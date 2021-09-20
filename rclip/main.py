import itertools
import os
from os import path
import re
from typing import Iterable, List, NamedTuple, Optional, Tuple, TypedDict, cast
import subprocess
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import PIL
from PIL import Image, ImageFile

import db, model, utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageMeta(TypedDict):
    modified_at: float
    size: int


def get_image_meta(filepath: str) -> ImageMeta:
    return ImageMeta(
        modified_at=os.path.getmtime(filepath),
        size=os.path.getsize(filepath)
    )


def is_image_meta_equal(image: db.Image, meta: ImageMeta) -> bool:
    for key in meta:
        if meta[key] != image[key]:
            return False
    return True


class RClip:
    EXCLUDE_DIRS_DEFAULT = ['@eaDir', 'node_modules', '.git']
    IMAGE_REGEX = re.compile(r'^.+\.(jpe?g|png)$', re.I)
    BATCH_SIZE = 8
    DB_IMAGES_BEFORE_COMMIT = 50_000

    class SearchResult(NamedTuple):
        filepath: str
        score: float

    def __init__(self, model_instance: model.Model, database: db.DB, exclude_dirs: Optional[List[str]]):
        self._model = model_instance
        self._db = database

        excluded_dirs = '|'.join(re.escape(dir) for dir in exclude_dirs or self.EXCLUDE_DIRS_DEFAULT)
        self._exclude_dir_regex = re.compile(f'^.+\\/({excluded_dirs})(\\/.+)?$')

    def _index_files(self, filepaths: List[str], metas: List[ImageMeta]):
        images: List[Image.Image] = []
        filtered_paths: List[str] = []
        for path in filepaths:
            try:
                image = Image.open(path)
                images.append(image)
                filtered_paths.append(path)
            except PIL.UnidentifiedImageError as ex:
                pass
            except Exception as ex:
                print(f'error loading image {path}:', ex)

        try:
            features = self._model.compute_image_features(images)
        except Exception as ex:
            print('error computing features:', ex)
            return
        for path, meta, vector in cast(Iterable[Tuple[str, ImageMeta, np.ndarray]],
                                       zip(filtered_paths, metas, features)):
            self._db.upsert_image(db.NewImage(
                filepath=path,
                modified_at=meta['modified_at'],
                size=meta['size'],
                vector=vector.tobytes()
            ), commit=False)

    def ensure_index(self, directory: str):
        # We will mark existing images as existing later
        self._db.flag_images_in_a_dir_as_deleted(directory)

        images_processed = 0
        batch: List[str] = []
        metas: List[ImageMeta] = []
        for root, _, files in os.walk(directory):
            if self._exclude_dir_regex.match(root):
                continue
            filtered_files = list(f for f in files if self.IMAGE_REGEX.match(f))
            if not filtered_files:
                continue
            for file in cast(Iterable[str], tqdm(filtered_files, desc=root)):
                filepath = path.join(root, file)

                image = self._db.get_image(filepath=filepath)
                try:
                    meta = get_image_meta(filepath)
                except Exception as ex:
                    print(f'error getting fs metadata for {filepath}:', ex)
                    continue

                if not images_processed % self.DB_IMAGES_BEFORE_COMMIT:
                    self._db.commit()
                images_processed += 1

                if image and is_image_meta_equal(image, meta):
                    self._db.remove_deleted_flag(filepath, commit=False)
                    continue

                batch.append(filepath)
                metas.append(meta)

                if len(batch) >= self.BATCH_SIZE:
                    self._index_files(batch, metas)
                    batch = []
                    metas = []

        if len(batch) != 0:
            self._index_files(batch, metas)

        self._db.commit()

    def search(self, query: str, directory: str, top_k: int = 10) -> List[SearchResult]:
        filepaths, features = self._get_features(directory)

        sorted_similarities = self._model.compute_similarities_to_text(features, query)

        filtered_similarities = filter(
            lambda similarity: not self._exclude_dir_regex.match(filepaths[similarity[1]]),
            sorted_similarities
        )
        top_k_similarities = itertools.islice(filtered_similarities, top_k)

        return [RClip.SearchResult(filepath=filepaths[th[1]], score=th[0]) for th in top_k_similarities]

    def _get_features(self, directory: str) -> Tuple[List[str], np.ndarray]:
        filepaths: List[str] = []
        features: List[np.ndarray] = []
        for image in self._db.get_image_vectors_by_dir_path(directory):
            filepaths.append(image['filepath'])
            features.append(np.frombuffer(image['vector'], np.float32))
        if not filepaths:
            return [], np.ndarray(shape=(0, model.Model.VECTOR_SIZE))
        return filepaths, np.stack(features)


def main():
    arg_parser = utils.init_arg_parser()
    args = arg_parser.parse_args()

    #current_directory = os.getcwd()
    current_directory = args.dir

    model_instance = model.Model()
    datadir = utils.get_app_datadir()
    database = db.DB(datadir / 'db.sqlite3')
    rclip = RClip(model_instance, database, args.exclude_dir)

    truncated_query = args.query[:141] # maximum limit for ecryptfs filenames/paths
    results_dir = os.path.join("results", truncated_query)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not args.skip_index:
        rclip.ensure_index(current_directory)

    result = rclip.search(args.query, current_directory, args.top)
    top_paths = []
    top_similarities = []
    if args.filepath_only:
        for r in result:
            print(r.filepath)
    else:
        with open(os.path.join(results_dir, 'topn'), 'w') as tF, open(os.path.join(results_dir, 'score'), 'w') as sF:
            print('score\tfilepath')
            for r in result:
                print(f'{r.score:.4f}\t"{r.filepath}"')
                tF.write(r.filepath+'\n')
                sF.write(f'{r.score:.4f}\n')

                top_paths.append(r.filepath)
                top_similarities.append(r.score)

    # show results in feh (non-blocking)
    proc = subprocess.Popen(['feh', '-t', '-f', './results/'+args.query+'/topn', '--no-screen-clip', '-E 200', '-y 200', '-d'])


    # plot similarities

    fig1 = utils.scatterImage(top_paths, top_similarities, zoom=0.12, size=(20, 5))
    plt.ioff()
    fig1.savefig(os.path.join(results_dir, 'results.svg'), dpi=fig1.dpi)

    fig3 = utils.scatterImage(top_paths, top_similarities, zoom=0.06, size=(20, 5))
    fig3.savefig(os.path.join(results_dir, 'results_zoomed.svg'), dpi=fig1.dpi)

    fig2 = utils.scatterImage(top_paths, top_similarities, zoom=0.08, size=(7, 12))
    # don't show:
    plt.close(fig3); plt.close(fig1)
    plt.show()
    fig2.savefig(os.path.join(results_dir, 'results_zoomed_flip.svg'))  # , dpi=400)

    # show results in feh (blocking)
    # subprocess.call(['feh', '-t', '-f', './results/'+args.query+'/topn', '--no-screen-clip', '-E 200', '-y 200', '-d'])

    # TODO save top-cluster images in results

    proc.wait() # require abort or closing feh to exit

if __name__ == '__main__':
    main()
