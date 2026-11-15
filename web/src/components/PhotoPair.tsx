import type { Photo, PhotoPair as PhotoPairData } from '../services/types';

function PhotoFrame({ photo }: { photo: Photo }) {
  // Only placeholders exist for now. A saved-photo kind will render an <img> from private storage.
  return (
    <div className="photo-placeholder" role="img" aria-label={photo.alt}>
      <span aria-hidden="true">Placeholder photo</span>
    </div>
  );
}

/** Matched evidence: the same spot at move-in and at move-out, side by side with clear labels. */
export function PhotoPair({ pair }: { pair: PhotoPairData }) {
  return (
    <div className="photo-pair">
      <figure className="photo-pair__item">
        <PhotoFrame photo={pair.moveIn} />
        <figcaption>
          <span className="photo-pair__label">Move-in</span>
          <span className="photo-pair__caption">{pair.moveIn.caption}</span>
        </figcaption>
      </figure>
      <figure className="photo-pair__item">
        <PhotoFrame photo={pair.moveOut} />
        <figcaption>
          <span className="photo-pair__label">Move-out</span>
          <span className="photo-pair__caption">{pair.moveOut.caption}</span>
        </figcaption>
      </figure>
    </div>
  );
}
