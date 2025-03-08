
@final
class CompressionCodec(Mapping[int, Callable[..., object]]):
    """Map :py:class:`COMPRESSION` value to encode or decode function.

    Parameters:
        encode: If *True*, return encode functions, else decode functions.

    """

    _codecs: dict[int, Callable[..., Any]]
    _encode: bool

    def __init__(self, encode: bool) -> None:
        self._codecs = {1: identityfunc}
        self._encode = bool(encode)

    def __getitem__(self, key: int, /) -> Callable[..., Any]:
        if key in self._codecs:
            return self._codecs[key]
        codec: Callable[..., Any]
        try:
            # TODO: enable CCITTRLE decoder for future imagecodecs
            # if key == 2:
            #     if self._encode:
            #         codec = imagecodecs.ccittrle_encode
            #     else:
            #         codec = imagecodecs.ccittrle_decode
            if key == 5:
                if self._encode:
                    codec = imagecodecs.lzw_encode
                else:
                    codec = imagecodecs.lzw_decode
            elif key in {6, 7, 33007}:
                if self._encode:
                    if key in {6, 33007}:
                        raise NotImplementedError
                    codec = imagecodecs.jpeg_encode
                else:
                    codec = imagecodecs.jpeg_decode
            elif key in {8, 32946, 50013}:
                if (
                    hasattr(imagecodecs, 'DEFLATE')
                    and imagecodecs.DEFLATE.available
                ):
                    # imagecodecs built with deflate
                    if self._encode:
                        codec = imagecodecs.deflate_encode
                    else:
                        codec = imagecodecs.deflate_decode
                elif (
                    hasattr(imagecodecs, 'ZLIB') and imagecodecs.ZLIB.available
                ):
                    if self._encode:
                        codec = imagecodecs.zlib_encode
                    else:
                        codec = imagecodecs.zlib_decode
                else:
                    # imagecodecs built without zlib
                    try:
                        from . import _imagecodecs
                    except ImportError:
                        import _imagecodecs  # type: ignore[no-redef]

                    if self._encode:
                        codec = _imagecodecs.zlib_encode
                    else:
                        codec = _imagecodecs.zlib_decode
            elif key == 32773:
                if self._encode:
                    codec = imagecodecs.packbits_encode
                else:
                    codec = imagecodecs.packbits_decode
            elif key in {33003, 33004, 33005, 34712}:
                if self._encode:
                    codec = imagecodecs.jpeg2k_encode
                else:
                    codec = imagecodecs.jpeg2k_decode
            elif key == 34887:
                if self._encode:
                    codec = imagecodecs.lerc_encode
                else:
                    codec = imagecodecs.lerc_decode
            elif key == 34892:
                # DNG lossy
                if self._encode:
                    codec = imagecodecs.jpeg8_encode
                else:
                    codec = imagecodecs.jpeg8_decode
            elif key == 34925:
                if hasattr(imagecodecs, 'LZMA') and imagecodecs.LZMA.available:
                    if self._encode:
                        codec = imagecodecs.lzma_encode
                    else:
                        codec = imagecodecs.lzma_decode
                else:
                    # imagecodecs built without lzma
                    try:
                        from . import _imagecodecs
                    except ImportError:
                        import _imagecodecs  # type: ignore[no-redef]

                    if self._encode:
                        codec = _imagecodecs.lzma_encode
                    else:
                        codec = _imagecodecs.lzma_decode
            elif key == 34933:
                if self._encode:
                    codec = imagecodecs.png_encode
                else:
                    codec = imagecodecs.png_decode
            elif key in {34934, 22610}:
                if self._encode:
                    codec = imagecodecs.jpegxr_encode
                else:
                    codec = imagecodecs.jpegxr_decode
            elif key == 48124:
                if self._encode:
                    codec = imagecodecs.jetraw_encode
                else:
                    codec = imagecodecs.jetraw_decode
            elif key in {50000, 34926}:  # 34926 deprecated
                if self._encode:
                    codec = imagecodecs.zstd_encode
                else:
                    codec = imagecodecs.zstd_decode
            elif key in {50001, 34927}:  # 34927 deprecated
                if self._encode:
                    codec = imagecodecs.webp_encode
                else:
                    codec = imagecodecs.webp_decode
            elif key in {65000, 65001, 65002} and not self._encode:
                codec = imagecodecs.eer_decode
            elif key in {50002, 52546}:
                if self._encode:
                    codec = imagecodecs.jpegxl_encode
                else:
                    codec = imagecodecs.jpegxl_decode
            else:
                try:
                    msg = f'{COMPRESSION(key)!r} not supported'
                except ValueError:
                    msg = f'{key} is not a known COMPRESSION'
                raise KeyError(msg)
        except (AttributeError, ImportError) as exc:
            raise KeyError(
                f'{COMPRESSION(key)!r} ' "requires the 'imagecodecs' package"
            ) from exc
        except NotImplementedError as exc:
            raise KeyError(f'{COMPRESSION(key)!r} not implemented') from exc
        self._codecs[key] = codec
        return codec

    def __contains__(self, key: Any, /) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield 1  # dummy

    def __len__(self) -> int:
        return 1  # dummy


@final
class PredictorCodec(Mapping[int, Callable[..., object]]):
    """Map :py:class:`PREDICTOR` value to encode or decode function.

    Parameters:
        encode: If *True*, return encode functions, else decode functions.

    """

    _codecs: dict[int, Callable[..., Any]]
    _encode: bool

    def __init__(self, encode: bool) -> None:
        self._codecs = {1: identityfunc}
        self._encode = bool(encode)

    def __getitem__(self, key: int, /) -> Callable[..., Any]:
        if key in self._codecs:
            return self._codecs[key]
        codec: Callable[..., Any]
        try:
            if key == 2:
                if self._encode:
                    codec = imagecodecs.delta_encode
                else:
                    codec = imagecodecs.delta_decode
            elif key == 3:
                if self._encode:
                    codec = imagecodecs.floatpred_encode
                else:
                    codec = imagecodecs.floatpred_decode
            elif key == 34892:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_encode(
                            data, axis=axis, out=out, dist=2
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_decode(
                            data, axis=axis, out=out, dist=2
                        )

            elif key == 34893:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_encode(
                            data, axis=axis, out=out, dist=4
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_decode(
                            data, axis=axis, out=out, dist=4
                        )

            elif key == 34894:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_encode(
                            data, axis=axis, out=out, dist=2
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_decode(
                            data, axis=axis, out=out, dist=2
                        )

            elif key == 34895:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_encode(
                            data, axis=axis, out=out, dist=4
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_decode(
                            data, axis=axis, out=out, dist=4
                        )

            else:
                raise KeyError(f'{key} is not a known PREDICTOR')
        except AttributeError as exc:
            raise KeyError(
                f'{PREDICTOR(key)!r}' " requires the 'imagecodecs' package"
            ) from exc
        except NotImplementedError as exc:
            raise KeyError(f'{PREDICTOR(key)!r} not implemented') from exc
        self._codecs[key] = codec
        return codec

    def __contains__(self, key: Any, /) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield 1  # dummy

    def __len__(self) -> int:
        return 1  # dummy

