import type { Photo, PhotoPair as PhotoPairData } from '../services/types';

function Region({ title, photo }: { title: string; photo: Photo }) {
  // Only placeholders exist for now. A saved-photo kind will render an <img> from private storage.
  return (
    <figure className="compare__region">
      <figcaption className="compare__label">{title}</figcaption>
      <div className="compare__frame" role="img" aria-label={photo.alt}>
        <span aria-hidden="true">No photo attached</span>
      </div>
    </figure>
  );
}

/** Matched evidence for one finding: the same spot at move-in and at move-out, side by side and compact. */
export function PhotoPair({ pair }: { pair: PhotoPairData }) {
  return (
    <div className="compare">
      <Region title="Move-in" photo={pair.moveIn} />
      <Region title="Move-out" photo={pair.moveOut} />
    </div>
  );
}
