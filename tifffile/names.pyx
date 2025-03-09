# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=True
# distutils: language=c++

from libc.stdint cimport int32_t


cdef object get_tag_names(int32_t tag):
    """Registry of TIFF tag codes and names from TIFF6, TIFF/EP, EXIF."""
    # This gets converted in a big switch statement
    if tag == 11:
        return 'ProcessingSoftware'
    elif tag == 254:
        return 'NewSubfileType'
    elif tag == 255:
        return 'SubfileType'
    elif tag == 256:
        return 'ImageWidth'
    elif tag == 257:
        return 'ImageLength'
    elif tag == 258:
        return 'BitsPerSample'
    elif tag == 259:
        return 'Compression'
    elif tag == 262:
        return 'PhotometricInterpretation'
    elif tag == 263:
        return 'Thresholding'
    elif tag == 264:
        return 'CellWidth'
    elif tag == 265:
        return 'CellLength'
    elif tag == 266:
        return 'FillOrder'
    elif tag == 269:
        return 'DocumentName'
    elif tag == 270:
        return 'ImageDescription'
    elif tag == 271:
        return 'Make'
    elif tag == 272:
        return 'Model'
    elif tag == 273:
        return 'StripOffsets'
    elif tag == 274:
        return 'Orientation'
    elif tag == 277:
        return 'SamplesPerPixel'
    elif tag == 278:
        return 'RowsPerStrip'
    elif tag == 279:
        return 'StripByteCounts'
    elif tag == 280:
        return 'MinSampleValue'
    elif tag == 281:
        return 'MaxSampleValue'
    elif tag == 282:
        return 'XResolution'
    elif tag == 283:
        return 'YResolution'
    elif tag == 284:
        return 'PlanarConfiguration'
    elif tag == 285:
        return 'PageName'
    elif tag == 286:
        return 'XPosition'
    elif tag == 287:
        return 'YPosition'
    elif tag == 288:
        return 'FreeOffsets'
    elif tag == 289:
        return 'FreeByteCounts'
    elif tag == 290:
        return 'GrayResponseUnit'
    elif tag == 291:
        return 'GrayResponseCurve'
    elif tag == 292:
        return 'T4Options'
    elif tag == 293:
        return 'T6Options'
    elif tag == 296:
        return 'ResolutionUnit'
    elif tag == 297:
        return 'PageNumber'
    elif tag == 300:
        return 'ColorResponseUnit'
    elif tag == 301:
        return 'TransferFunction'
    elif tag == 305:
        return 'Software'
    elif tag == 306:
        return 'DateTime'
    elif tag == 315:
        return 'Artist'
    elif tag == 316:
        return 'HostComputer'
    elif tag == 317:
        return 'Predictor'
    elif tag == 318:
        return 'WhitePoint'
    elif tag == 319:
        return 'PrimaryChromaticities'
    elif tag == 320:
        return 'ColorMap'
    elif tag == 321:
        return 'HalftoneHints'
    elif tag == 322:
        return 'TileWidth'
    elif tag == 323:
        return 'TileLength'
    elif tag == 324:
        return 'TileOffsets'
    elif tag == 325:
        return 'TileByteCounts'
    elif tag == 326:
        return 'BadFaxLines'
    elif tag == 327:
        return 'CleanFaxData'
    elif tag == 328:
        return 'ConsecutiveBadFaxLines'
    elif tag == 330:
        return 'SubIFDs'
    elif tag == 332:
        return 'InkSet'
    elif tag == 333:
        return 'InkNames'
    elif tag == 334:
        return 'NumberOfInks'
    elif tag == 336:
        return 'DotRange'
    elif tag == 337:
        return 'TargetPrinter'
    elif tag == 338:
        return 'ExtraSamples'
    elif tag == 339:
        return 'SampleFormat'
    elif tag == 340:
        return 'SMinSampleValue'
    elif tag == 341:
        return 'SMaxSampleValue'
    elif tag == 342:
        return 'TransferRange'
    elif tag == 343:
        return 'ClipPath'
    elif tag == 344:
        return 'XClipPathUnits'
    elif tag == 345:
        return 'YClipPathUnits'
    elif tag == 346:
        return 'Indexed'
    elif tag == 347:
        return 'JPEGTables'
    elif tag == 351:
        return 'OPIProxy'
    elif tag == 400:
        return 'GlobalParametersIFD'
    elif tag == 401:
        return 'ProfileType'
    elif tag == 402:
        return 'FaxProfile'
    elif tag == 403:
        return 'CodingMethods'
    elif tag == 404:
        return 'VersionYear'
    elif tag == 405:
        return 'ModeNumber'
    elif tag == 433:
        return 'Decode'
    elif tag == 434:
        return 'DefaultImageColor'
    elif tag == 435:
        return 'T82Options'
    elif tag == 437:
        return 'JPEGTables'  # 347
    elif tag == 512:
        return 'JPEGProc'
    elif tag == 513:
        return 'JPEGInterchangeFormat'
    elif tag == 514:
        return 'JPEGInterchangeFormatLength'
    elif tag == 515:
        return 'JPEGRestartInterval'
    elif tag == 517:
        return 'JPEGLosslessPredictors'
    elif tag == 518:
        return 'JPEGPointTransforms'
    elif tag == 519:
        return 'JPEGQTables'
    elif tag == 520:
        return 'JPEGDCTables'
    elif tag == 521:
        return 'JPEGACTables'
    elif tag == 529:
        return 'YCbCrCoefficients'
    elif tag == 530:
        return 'YCbCrSubSampling'
    elif tag == 531:
        return 'YCbCrPositioning'
    elif tag == 532:
        return 'ReferenceBlackWhite'
    elif tag == 559:
        return 'StripRowCounts'
    elif tag == 700:
        return 'XMP'  # XMLPacket
    elif tag == 769:
        return 'GDIGamma'  # GDI+
    elif tag == 770:
        return 'ICCProfileDescriptor'  # GDI+
    elif tag == 771:
        return 'SRGBRenderingIntent'  # GDI+
    elif tag == 800:
        return 'ImageTitle'  # GDI+
    elif tag == 907:
        return 'SiffCompress'  # https://github.com/MaimonLab/SiffPy
    elif tag == 999:
        return 'USPTO_Miscellaneous'
    elif tag == 4864:
        return 'AndorId'  # TODO, Andor Technology 4864 - 5030
    elif tag == 4869:
        return 'AndorTemperature'
    elif tag == 4876:
        return 'AndorExposureTime'
    elif tag == 4878:
        return 'AndorKineticCycleTime'
    elif tag == 4879:
        return 'AndorAccumulations'
    elif tag == 4881:
        return 'AndorAcquisitionCycleTime'
    elif tag == 4882:
        return 'AndorReadoutTime'
    elif tag == 4884:
        return 'AndorPhotonCounting'
    elif tag == 4885:
        return 'AndorEmDacLevel'
    elif tag == 4890:
        return 'AndorFrames'
    elif tag == 4896:
        return 'AndorHorizontalFlip'
    elif tag == 4897:
        return 'AndorVerticalFlip'
    elif tag == 4898:
        return 'AndorClockwise'
    elif tag == 4899:
        return 'AndorCounterClockwise'
    elif tag == 4904:
        return 'AndorVerticalClockVoltage'
    elif tag == 4905:
        return 'AndorVerticalShiftSpeed'
    elif tag == 4907:
        return 'AndorPreAmpSetting'
    elif tag == 4908:
        return 'AndorCameraSerial'
    elif tag == 4911:
        return 'AndorActualTemperature'
    elif tag == 4912:
        return 'AndorBaselineClamp'
    elif tag == 4913:
        return 'AndorPrescans'
    elif tag == 4914:
        return 'AndorModel'
    elif tag == 4915:
        return 'AndorChipSizeX'
    elif tag == 4916:
        return 'AndorChipSizeY'
    elif tag == 4944:
        return 'AndorBaselineOffset'
    elif tag == 4966:
        return 'AndorSoftwareVersion'
    elif tag == 18246:
        return 'Rating'
    elif tag == 18247:
        return 'XP_DIP_XML'
    elif tag == 18248:
        return 'StitchInfo'
    elif tag == 18249:
        return 'RatingPercent'
    elif tag == 20481:
        return 'ResolutionXUnit'  # GDI+
    elif tag == 20482:
        return 'ResolutionYUnit'  # GDI+
    elif tag == 20483:
        return 'ResolutionXLengthUnit'  # GDI+
    elif tag == 20484:
        return 'ResolutionYLengthUnit'  # GDI+
    elif tag == 20485:
        return 'PrintFlags'  # GDI+
    elif tag == 20486:
        return 'PrintFlagsVersion'  # GDI+
    elif tag == 20487:
        return 'PrintFlagsCrop'  # GDI+
    elif tag == 20488:
        return 'PrintFlagsBleedWidth'  # GDI+
    elif tag == 20489:
        return 'PrintFlagsBleedWidthScale'  # GDI+
    elif tag == 20490:
        return 'HalftoneLPI'  # GDI+
    elif tag == 20491:
        return 'HalftoneLPIUnit'  # GDI+
    elif tag == 20492:
        return 'HalftoneDegree'  # GDI+
    elif tag == 20493:
        return 'HalftoneShape'  # GDI+
    elif tag == 20494:
        return 'HalftoneMisc'  # GDI+
    elif tag == 20495:
        return 'HalftoneScreen'  # GDI+
    elif tag == 20496:
        return 'JPEGQuality'  # GDI+
    elif tag == 20497:
        return 'GridSize'  # GDI+
    elif tag == 20498:
        return 'ThumbnailFormat'  # GDI+
    elif tag == 20499:
        return 'ThumbnailWidth'  # GDI+
    elif tag == 20500:
        return 'ThumbnailHeight'  # GDI+
    elif tag == 20501:
        return 'ThumbnailColorDepth'  # GDI+
    elif tag == 20502:
        return 'ThumbnailPlanes'  # GDI+
    elif tag == 20503:
        return 'ThumbnailRawBytes'  # GDI+
    elif tag == 20504:
        return 'ThumbnailSize'  # GDI+
    elif tag == 20505:
        return 'ThumbnailCompressedSize'  # GDI+
    elif tag == 20506:
        return 'ColorTransferFunction'  # GDI+
    elif tag == 20507:
        return 'ThumbnailData'
    elif tag == 20512:
        return 'ThumbnailImageWidth'  # GDI+
    elif tag == 20513:
        return 'ThumbnailImageHeight'  # GDI+
    elif tag == 20514:
        return 'ThumbnailBitsPerSample'  # GDI+
    elif tag == 20515:
        return 'ThumbnailCompression'
    elif tag == 20516:
        return 'ThumbnailPhotometricInterp'  # GDI+
    elif tag == 20517:
        return 'ThumbnailImageDescription'  # GDI+
    elif tag == 20518:
        return 'ThumbnailEquipMake'  # GDI+
    elif tag == 20519:
        return 'ThumbnailEquipModel'  # GDI+
    elif tag == 20520:
        return 'ThumbnailStripOffsets'  # GDI+
    elif tag == 20521:
        return 'ThumbnailOrientation'  # GDI+
    elif tag == 20522:
        return 'ThumbnailSamplesPerPixel'  # GDI+
    elif tag == 20523:
        return 'ThumbnailRowsPerStrip'  # GDI+
    elif tag == 20524:
        return 'ThumbnailStripBytesCount'  # GDI+
    elif tag == 20525:
        return 'ThumbnailResolutionX'
    elif tag == 20526:
        return 'ThumbnailResolutionY'
    elif tag == 20527:
        return 'ThumbnailPlanarConfig'  # GDI+
    elif tag == 20528:
        return 'ThumbnailResolutionUnit'
    elif tag == 20529:
        return 'ThumbnailTransferFunction'
    elif tag == 20530:
        return 'ThumbnailSoftwareUsed'  # GDI+
    elif tag == 20531:
        return 'ThumbnailDateTime'  # GDI+
    elif tag == 20532:
        return 'ThumbnailArtist'  # GDI+
    elif tag == 20533:
        return 'ThumbnailWhitePoint'  # GDI+
    elif tag == 20534:
        return 'ThumbnailPrimaryChromaticities'  # GDI+
    elif tag == 20535:
        return 'ThumbnailYCbCrCoefficients'  # GDI+
    elif tag == 20536:
        return 'ThumbnailYCbCrSubsampling'  # GDI+
    elif tag == 20537:
        return 'ThumbnailYCbCrPositioning'
    elif tag == 20538:
        return 'ThumbnailRefBlackWhite'  # GDI+
    elif tag == 20539:
        return 'ThumbnailCopyRight'  # GDI+
    elif tag == 20545:
        return 'InteroperabilityIndex'
    elif tag == 20546:
        return 'InteroperabilityVersion'
    elif tag == 20624:
        return 'LuminanceTable'
    elif tag == 20625:
        return 'ChrominanceTable'
    elif tag == 20736:
        return 'FrameDelay'  # GDI+
    elif tag == 20737:
        return 'LoopCount'  # GDI+
    elif tag == 20738:
        return 'GlobalPalette'  # GDI+
    elif tag == 20739:
        return 'IndexBackground'  # GDI+
    elif tag == 20740:
        return 'IndexTransparent'  # GDI+
    elif tag == 20752:
        return 'PixelUnit'  # GDI+
    elif tag == 20753:
        return 'PixelPerUnitX'  # GDI+
    elif tag == 20754:
        return 'PixelPerUnitY'  # GDI+
    elif tag == 20755:
        return 'PaletteHistogram'  # GDI+
    elif tag == 28672:
        return 'SonyRawFileType'  # Sony ARW
    elif tag == 28722:
        return 'VignettingCorrParams'  # Sony ARW
    elif tag == 28725:
        return 'ChromaticAberrationCorrParams'  # Sony ARW
    elif tag == 28727:
        return 'DistortionCorrParams'  # Sony ARW
                # Private tags >= 32768
    elif tag == 32781:
        return 'ImageID'
    elif tag == 32931:
        return 'WangTag1'
    elif tag == 32932:
        return 'WangAnnotation'
    elif tag == 32933:
        return 'WangTag3'
    elif tag == 32934:
        return 'WangTag4'
    elif tag == 32953:
        return 'ImageReferencePoints'
    elif tag == 32954:
        return 'RegionXformTackPoint'
    elif tag == 32955:
        return 'WarpQuadrilateral'
    elif tag == 32956:
        return 'AffineTransformMat'
    elif tag == 32995:
        return 'Matteing'
    elif tag == 32996:
        return 'DataType'  # use SampleFormat
    elif tag == 32997:
        return 'ImageDepth'
    elif tag == 32998:
        return 'TileDepth'
    elif tag == 33300:
        return 'ImageFullWidth'
    elif tag == 33301:
        return 'ImageFullLength'
    elif tag == 33302:
        return 'TextureFormat'
    elif tag == 33303:
        return 'TextureWrapModes'
    elif tag == 33304:
        return 'FieldOfViewCotangent'
    elif tag == 33305:
        return 'MatrixWorldToScreen'
    elif tag == 33306:
        return 'MatrixWorldToCamera'
    elif tag == 33405:
        return 'Model2'
    elif tag == 33421:
        return 'CFARepeatPatternDim'
    elif tag == 33422:
        return 'CFAPattern'
    elif tag == 33423:
        return 'BatteryLevel'
    elif tag == 33424:
        return 'KodakIFD'
    elif tag == 33432:
        return 'Copyright'
    elif tag == 33434:
        return 'ExposureTime'
    elif tag == 33437:
        return 'FNumber'
    elif tag == 33445:
        return 'MDFileTag'
    elif tag == 33446:
        return 'MDScalePixel'
    elif tag == 33447:
        return 'MDColorTable'
    elif tag == 33448:
        return 'MDLabName'
    elif tag == 33449:
        return 'MDSampleInfo'
    elif tag == 33450:
        return 'MDPrepDate'
    elif tag == 33451:
        return 'MDPrepTime'
    elif tag == 33452:
        return 'MDFileUnits'
    elif tag == 33465:
        return 'NiffRotation'  # NIFF
    elif tag == 33466:
        return 'NiffNavyCompression'  # NIFF
    elif tag == 33467:
        return 'NiffTileIndex'  # NIFF
    elif tag == 33471:
        return 'OlympusINI'
    elif tag == 33550:
        return 'ModelPixelScaleTag'
    elif tag == 33560:
        return 'OlympusSIS'  # see also 33471 and 34853
    elif tag == 33589:
        return 'AdventScale'
    elif tag == 33590:
        return 'AdventRevision'
    elif tag == 33628:
        return 'UIC1tag'  # Metamorph  Universal Imaging Corp STK
    elif tag == 33629:
        return 'UIC2tag'
    elif tag == 33630:
        return 'UIC3tag'
    elif tag == 33631:
        return 'UIC4tag'
    elif tag == 33723:
        return 'IPTCNAA'
    elif tag == 33858:
        return 'ExtendedTagsOffset'  # DEFF points IFD with tags
    elif tag == 33918:
        return 'IntergraphPacketData'  # INGRPacketDataTag
    elif tag == 33919:
        return 'IntergraphFlagRegisters'  # INGRFlagRegisters
    elif tag == 33920:
        return 'IntergraphMatrixTag'  # IrasBTransformationMatrix
    elif tag == 33921:
        return 'INGRReserved'
    elif tag == 33922:
        return 'ModelTiepointTag'
    elif tag == 33923:
        return 'LeicaMagic'
    elif tag == 34016:
        return 'Site'  # 34016..34032 ANSI IT8 TIFF/IT
    elif tag == 34017:
        return 'ColorSequence'
    elif tag == 34018:
        return 'IT8Header'
    elif tag == 34019:
        return 'RasterPadding'
    elif tag == 34020:
        return 'BitsPerRunLength'
    elif tag == 34021:
        return 'BitsPerExtendedRunLength'
    elif tag == 34022:
        return 'ColorTable'
    elif tag == 34023:
        return 'ImageColorIndicator'
    elif tag == 34024:
        return 'BackgroundColorIndicator'
    elif tag == 34025:
        return 'ImageColorValue'
    elif tag == 34026:
        return 'BackgroundColorValue'
    elif tag == 34027:
        return 'PixelIntensityRange'
    elif tag == 34028:
        return 'TransparencyIndicator'
    elif tag == 34029:
        return 'ColorCharacterization'
    elif tag == 34030:
        return 'HCUsage'
    elif tag == 34031:
        return 'TrapIndicator'
    elif tag == 34032:
        return 'CMYKEquivalent'
    elif tag == 34118:
        return 'CZ_SEM'  # Zeiss SEM
    elif tag == 34122:
        return 'IPLAB'  # number of images
    elif tag == 34152:
        return 'AFCP_IPTC'
    elif tag == 34232:
        return 'PixelMagicJBIGOptions'  # EXIF, also TI FrameCount
    elif tag == 34263:
        return 'JPLCartoIFD'
    elif tag == 34264:
        return 'ModelTransformationTag'
    elif tag == 34306:
        return 'WB_GRGBLevels'  # Leaf MOS
    elif tag == 34310:
        return 'LeafData'
    elif tag == 34361:
        return 'MM_Header'
    elif tag == 34362:
        return 'MM_Stamp'
    elif tag == 34363:
        return 'MM_Unknown'
    elif tag == 34377:
        return 'ImageResources'  # Photoshop
    elif tag == 34386:
        return 'MM_UserBlock'
    elif tag == 34412:
        return 'CZ_LSMINFO'
    elif tag == 34665:
        return 'ExifTag'
    elif tag == 34675:
        return 'InterColorProfile'  # ICCProfile
    elif tag == 34680:
        return 'FEI_SFEG'  #
    elif tag == 34682:
        return 'FEI_HELIOS'  #
    elif tag == 34683:
        return 'FEI_TITAN'  #
    elif tag == 34687:
        return 'FXExtensions'
    elif tag == 34688:
        return 'MultiProfiles'
    elif tag == 34689:
        return 'SharedData'
    elif tag == 34690:
        return 'T88Options'
    elif tag == 34710:
        return 'MarCCD'  # offset to MarCCD header
    elif tag == 34732:
        return 'ImageLayer'
    elif tag == 34735:
        return 'GeoKeyDirectoryTag'
    elif tag == 34736:
        return 'GeoDoubleParamsTag'
    elif tag == 34737:
        return 'GeoAsciiParamsTag'
    elif tag == 34750:
        return 'JBIGOptions'
    elif tag == 34821:
        return 'PIXTIFF'  # ? Pixel Translations Inc
    elif tag == 34850:
        return 'ExposureProgram'
    elif tag == 34852:
        return 'SpectralSensitivity'
    elif tag == 34853:
        return 'GPSTag'  # GPSIFD  also OlympusSIS2
    elif tag == 34855:
        return ('ISOSpeedRatings', 'PhotographicSensitivity')
    elif tag == 34856:
        return 'OECF'  # optoelectric conversion factor
    elif tag == 34857:
        return 'Interlace'  # TIFF/EP
    elif tag == 34858:
        return 'TimeZoneOffset'  # TIFF/EP
    elif tag == 34859:
        return 'SelfTimerMode'  # TIFF/EP
    elif tag == 34864:
        return 'SensitivityType'
    elif tag == 34865:
        return 'StandardOutputSensitivity'
    elif tag == 34866:
        return 'RecommendedExposureIndex'
    elif tag == 34867:
        return 'ISOSpeed'
    elif tag == 34868:
        return 'ISOSpeedLatitudeyyy'
    elif tag == 34869:
        return 'ISOSpeedLatitudezzz'
    elif tag == 34908:
        return 'HylaFAXFaxRecvParams'
    elif tag == 34909:
        return 'HylaFAXFaxSubAddress'
    elif tag == 34910:
        return 'HylaFAXFaxRecvTime'
    elif tag == 34911:
        return 'FaxDcs'
    elif tag == 34929:
        return 'FedexEDR'
    elif tag == 34954:
        return 'LeafSubIFD'
    elif tag == 34959:
        return 'Aphelion1'
    elif tag == 34960:
        return 'Aphelion2'
    elif tag == 34961:
        return 'AphelionInternal'  # ADCIS
    elif tag == 36864:
        return ('ExifVersion', 'TVX_Unknown')
    elif tag == 36865:
        return 'TVX_NumExposure'
    elif tag == 36866:
        return 'TVX_NumBackground'
    elif tag == 36867:
        return ('DateTimeOriginal', 'TVX_ExposureTime')
    elif tag == 36868:
        return ('DateTimeDigitized', 'TVX_BackgroundTime')
    elif tag == 36870:
        return 'TVX_Unknown'
    elif tag == 36873:
        return ('GooglePlusUploadCode', 'TVX_SubBpp')
    elif tag == 36874:
        return 'TVX_SubWide'
    elif tag == 36875:
        return 'TVX_SubHigh'
    elif tag == 36876:
        return 'TVX_BlackLevel'
    elif tag == 36877:
        return 'TVX_DarkCurrent'
    elif tag == 36878:
        return 'TVX_ReadNoise'
    elif tag == 36879:
        return 'TVX_DarkCurrentNoise'
    elif tag == 36880:
        return ('OffsetTime', 'TVX_BeamMonitor')
    elif tag == 36881:
        return 'OffsetTimeOriginal'
    elif tag == 36882:
        return 'OffsetTimeDigitized'
                # TODO, Pilatus/CHESS/TV6 36864..37120 conflicting with Exif
    elif tag == 37120:
        return 'TVX_UserVariables'  # A/D values
    elif tag == 37121:
        return 'ComponentsConfiguration'
    elif tag == 37122:
        return 'CompressedBitsPerPixel'
    elif tag == 37377:
        return 'ShutterSpeedValue'
    elif tag == 37378:
        return 'ApertureValue'
    elif tag == 37379:
        return 'BrightnessValue'
    elif tag == 37380:
        return 'ExposureBiasValue'
    elif tag == 37381:
        return 'MaxApertureValue'
    elif tag == 37382:
        return 'SubjectDistance'
    elif tag == 37383:
        return 'MeteringMode'
    elif tag == 37384:
        return 'LightSource'
    elif tag == 37385:
        return 'Flash'
    elif tag == 37386:
        return 'FocalLength'
    elif tag == 37387:
        return 'FlashEnergy'  # TIFF/EP
    elif tag == 37388:
        return 'SpatialFrequencyResponse'  # TIFF/EP
    elif tag == 37389:
        return 'Noise'  # TIFF/EP
    elif tag == 37390:
        return 'FocalPlaneXResolution'  # TIFF/EP
    elif tag == 37391:
        return 'FocalPlaneYResolution'  # TIFF/EP
    elif tag == 37392:
        return 'FocalPlaneResolutionUnit'  # TIFF/EP
    elif tag == 37393:
        return 'ImageNumber'  # TIFF/EP
    elif tag == 37394:
        return 'SecurityClassification'  # TIFF/EP
    elif tag == 37395:
        return 'ImageHistory'  # TIFF/EP
    elif tag == 37396:
        return 'SubjectLocation'  # TIFF/EP
    elif tag == 37397:
        return 'ExposureIndex'  # TIFF/EP
    elif tag == 37398:
        return 'TIFFEPStandardID'  # TIFF/EP
    elif tag == 37399:
        return 'SensingMethod'  # TIFF/EP
    elif tag == 37434:
        return 'CIP3DataFile'
    elif tag == 37435:
        return 'CIP3Sheet'
    elif tag == 37436:
        return 'CIP3Side'
    elif tag == 37439:
        return 'StoNits'
    elif tag == 37500:
        return 'MakerNote'
    elif tag == 37510:
        return 'UserComment'
    elif tag == 37520:
        return 'SubsecTime'
    elif tag == 37521:
        return 'SubsecTimeOriginal'
    elif tag == 37522:
        return 'SubsecTimeDigitized'
    elif tag == 37679:
        return 'MODIText'  # Microsoft Office Document Imaging
    elif tag == 37680:
        return 'MODIOLEPropertySetStorage'
    elif tag == 37681:
        return 'MODIPositioning'
    elif tag == 37701:
        return 'AgilentBinary'  # private structure
    elif tag == 37702:
        return 'AgilentString'  # file description
    elif tag == 37706:
        return 'TVIPS'  # offset to TemData structure
    elif tag == 37707:
        return 'TVIPS1'
    elif tag == 37708:
        return 'TVIPS2'  # same TemData structure as undefined
    elif tag == 37724:
        return 'ImageSourceData'  # Photoshop
    elif tag == 37888:
        return 'Temperature'
    elif tag == 37889:
        return 'Humidity'
    elif tag == 37890:
        return 'Pressure'
    elif tag == 37891:
        return 'WaterDepth'
    elif tag == 37892:
        return 'Acceleration'
    elif tag == 37893:
        return 'CameraElevationAngle'
    elif tag == 40000:
        return 'XPos'  # Janelia
    elif tag == 40001:
        return ('YPos', 'RecipName', 'MC_IpWinScal')  # MS FAX, Media Cybernetics
    elif tag == 40002:
        return ('ZPos', 'RecipNumber')
    elif tag == 40003:
        return 'SenderName'
    elif tag == 40004:
        return 'Routing'
    elif tag == 40005:
        return 'CallerId'
    elif tag == 40006:
        return 'TSID'
    elif tag == 40007:
        return 'CSID'
    elif tag == 40008:
        return 'FaxTime'
    elif tag == 40091:
        return 'XPTitle'
    elif tag == 40092:
        return 'XPComment'
    elif tag == 40093:
        return 'XPAuthor'
    elif tag == 40094:
        return 'XPKeywords'
    elif tag == 40095:
        return 'XPSubject'
    elif tag == 40100:
        return 'MC_IdOld'
    elif tag == 40106:
        return 'MC_Unknown'
    elif tag == 40960:
        return 'FlashpixVersion'
    elif tag == 40961:
        return 'ColorSpace'
    elif tag == 40962:
        return 'PixelXDimension'
    elif tag == 40963:
        return 'PixelYDimension'
    elif tag == 40964:
        return 'RelatedSoundFile'
    elif tag == 40965:
        return 'InteroperabilityTag'  # InteropOffset
    elif tag == 40976:
        return 'SamsungRawPointersOffset'
    elif tag == 40977:
        return 'SamsungRawPointersLength'
    elif tag == 41217:
        return 'SamsungRawByteOrder'
    elif tag == 41218:
        return 'SamsungRawUnknown'
    elif tag == 41483:
        return 'FlashEnergy'
    elif tag == 41484:
        return 'SpatialFrequencyResponse'
    elif tag == 41485:
        return 'Noise'  # 37389
    elif tag == 41486:
        return 'FocalPlaneXResolution'  # 37390
    elif tag == 41487:
        return 'FocalPlaneYResolution'  # 37391
    elif tag == 41488:
        return 'FocalPlaneResolutionUnit'  # 37392
    elif tag == 41489:
        return 'ImageNumber'  # 37393
    elif tag == 41490:
        return 'SecurityClassification'  # 37394
    elif tag == 41491:
        return 'ImageHistory'  # 37395
    elif tag == 41492:
        return 'SubjectLocation'  # 37395
    elif tag == 41493:
        return 'ExposureIndex '  # 37397
    elif tag == 41494:
        return 'TIFF-EPStandardID'
    elif tag == 41495:
        return 'SensingMethod'  # 37399
    elif tag == 41728:
        return 'FileSource'
    elif tag == 41729:
        return 'SceneType'
    elif tag == 41730:
        return 'CFAPattern'  # 33422
    elif tag == 41985:
        return 'CustomRendered'
    elif tag == 41986:
        return 'ExposureMode'
    elif tag == 41987:
        return 'WhiteBalance'
    elif tag == 41988:
        return 'DigitalZoomRatio'
    elif tag == 41989:
        return 'FocalLengthIn35mmFilm'
    elif tag == 41990:
        return 'SceneCaptureType'
    elif tag == 41991:
        return 'GainControl'
    elif tag == 41992:
        return 'Contrast'
    elif tag == 41993:
        return 'Saturation'
    elif tag == 41994:
        return 'Sharpness'
    elif tag == 41995:
        return 'DeviceSettingDescription'
    elif tag == 41996:
        return 'SubjectDistanceRange'
    elif tag == 42016:
        return 'ImageUniqueID'
    elif tag == 42032:
        return 'CameraOwnerName'
    elif tag == 42033:
        return 'BodySerialNumber'
    elif tag == 42034:
        return 'LensSpecification'
    elif tag == 42035:
        return 'LensMake'
    elif tag == 42036:
        return 'LensModel'
    elif tag == 42037:
        return 'LensSerialNumber'
    elif tag == 42080:
        return 'CompositeImage'
    elif tag == 42081:
        return 'SourceImageNumberCompositeImage'
    elif tag == 42082:
        return 'SourceExposureTimesCompositeImage'
    elif tag == 42112:
        return 'GDAL_METADATA'
    elif tag == 42113:
        return 'GDAL_NODATA'
    elif tag == 42240:
        return 'Gamma'
    elif tag == 43314:
        return 'NIHImageHeader'
    elif tag == 44992:
        return 'ExpandSoftware'
    elif tag == 44993:
        return 'ExpandLens'
    elif tag == 44994:
        return 'ExpandFilm'
    elif tag == 44995:
        return 'ExpandFilterLens'
    elif tag == 44996:
        return 'ExpandScanner'
    elif tag == 44997:
        return 'ExpandFlashLamp'
    elif tag == 48129:
        return 'PixelFormat'  # HDP and WDP
    elif tag == 48130:
        return 'Transformation'
    elif tag == 48131:
        return 'Uncompressed'
    elif tag == 48132:
        return 'ImageType'
    elif tag == 48256:
        return 'ImageWidth'  # 256
    elif tag == 48257:
        return 'ImageHeight'
    elif tag == 48258:
        return 'WidthResolution'
    elif tag == 48259:
        return 'HeightResolution'
    elif tag == 48320:
        return 'ImageOffset'
    elif tag == 48321:
        return 'ImageByteCount'
    elif tag == 48322:
        return 'AlphaOffset'
    elif tag == 48323:
        return 'AlphaByteCount'
    elif tag == 48324:
        return 'ImageDataDiscard'
    elif tag == 48325:
        return 'AlphaDataDiscard'
    elif tag == 50003:
        return 'KodakAPP3'
    elif tag == 50215:
        return 'OceScanjobDescription'
    elif tag == 50216:
        return 'OceApplicationSelector'
    elif tag == 50217:
        return 'OceIdentificationNumber'
    elif tag == 50218:
        return 'OceImageLogicCharacteristics'
    elif tag == 50255:
        return 'Annotations'
    elif tag == 50288:
        return 'MC_Id'  # Media Cybernetics
    elif tag == 50289:
        return 'MC_XYPosition'
    elif tag == 50290:
        return 'MC_ZPosition'
    elif tag == 50291:
        return 'MC_XYCalibration'
    elif tag == 50292:
        return 'MC_LensCharacteristics'
    elif tag == 50293:
        return 'MC_ChannelName'
    elif tag == 50294:
        return 'MC_ExcitationWavelength'
    elif tag == 50295:
        return 'MC_TimeStamp'
    elif tag == 50296:
        return 'MC_FrameProperties'
    elif tag == 50341:
        return 'PrintImageMatching'
    elif tag == 50495:
        return 'PCO_RAW'  # TODO, PCO CamWare
    elif tag == 50547:
        return 'OriginalFileName'
    elif tag == 50560:
        return 'USPTO_OriginalContentType'  # US Patent Office
    elif tag == 50561:
        return 'USPTO_RotationCode'
    elif tag == 50648:
        return 'CR2Unknown1'
    elif tag == 50649:
        return 'CR2Unknown2'
    elif tag == 50656:
        return 'CR2CFAPattern'
    elif tag == 50674:
        return 'LercParameters'  # ESGI 50674 .. 50677
    elif tag == 50706:
        return 'DNGVersion'  # DNG 50706 .. 51114
    elif tag == 50707:
        return 'DNGBackwardVersion'
    elif tag == 50708:
        return 'UniqueCameraModel'
    elif tag == 50709:
        return 'LocalizedCameraModel'
    elif tag == 50710:
        return 'CFAPlaneColor'
    elif tag == 50711:
        return 'CFALayout'
    elif tag == 50712:
        return 'LinearizationTable'
    elif tag == 50713:
        return 'BlackLevelRepeatDim'
    elif tag == 50714:
        return 'BlackLevel'
    elif tag == 50715:
        return 'BlackLevelDeltaH'
    elif tag == 50716:
        return 'BlackLevelDeltaV'
    elif tag == 50717:
        return 'WhiteLevel'
    elif tag == 50718:
        return 'DefaultScale'
    elif tag == 50719:
        return 'DefaultCropOrigin'
    elif tag == 50720:
        return 'DefaultCropSize'
    elif tag == 50721:
        return 'ColorMatrix1'
    elif tag == 50722:
        return 'ColorMatrix2'
    elif tag == 50723:
        return 'CameraCalibration1'
    elif tag == 50724:
        return 'CameraCalibration2'
    elif tag == 50725:
        return 'ReductionMatrix1'
    elif tag == 50726:
        return 'ReductionMatrix2'
    elif tag == 50727:
        return 'AnalogBalance'
    elif tag == 50728:
        return 'AsShotNeutral'
    elif tag == 50729:
        return 'AsShotWhiteXY'
    elif tag == 50730:
        return 'BaselineExposure'
    elif tag == 50731:
        return 'BaselineNoise'
    elif tag == 50732:
        return 'BaselineSharpness'
    elif tag == 50733:
        return 'BayerGreenSplit'
    elif tag == 50734:
        return 'LinearResponseLimit'
    elif tag == 50735:
        return 'CameraSerialNumber'
    elif tag == 50736:
        return 'LensInfo'
    elif tag == 50737:
        return 'ChromaBlurRadius'
    elif tag == 50738:
        return 'AntiAliasStrength'
    elif tag == 50739:
        return 'ShadowScale'
    elif tag == 50740:
        return 'DNGPrivateData'
    elif tag == 50741:
        return 'MakerNoteSafety'
    elif tag == 50752:
        return 'RawImageSegmentation'
    elif tag == 50778:
        return 'CalibrationIlluminant1'
    elif tag == 50779:
        return 'CalibrationIlluminant2'
    elif tag == 50780:
        return 'BestQualityScale'
    elif tag == 50781:
        return 'RawDataUniqueID'
    elif tag == 50784:
        return 'AliasLayerMetadata'
    elif tag == 50827:
        return 'OriginalRawFileName'
    elif tag == 50828:
        return 'OriginalRawFileData'
    elif tag == 50829:
        return 'ActiveArea'
    elif tag == 50830:
        return 'MaskedAreas'
    elif tag == 50831:
        return 'AsShotICCProfile'
    elif tag == 50832:
        return 'AsShotPreProfileMatrix'
    elif tag == 50833:
        return 'CurrentICCProfile'
    elif tag == 50834:
        return 'CurrentPreProfileMatrix'
    elif tag == 50838:
        return 'IJMetadataByteCounts'
    elif tag == 50839:
        return 'IJMetadata'
    elif tag == 50844:
        return 'RPCCoefficientTag'
    elif tag == 50879:
        return 'ColorimetricReference'
    elif tag == 50885:
        return 'SRawType'
    elif tag == 50898:
        return 'PanasonicTitle'
    elif tag == 50899:
        return 'PanasonicTitle2'
    elif tag == 50908:
        return 'RSID'  # DGIWG
    elif tag == 50909:
        return 'GEO_METADATA'  # DGIWG XML
    elif tag == 50931:
        return 'CameraCalibrationSignature'
    elif tag == 50932:
        return 'ProfileCalibrationSignature'
    elif tag == 50933:
        return 'ProfileIFD'  # EXTRACAMERAPROFILES
    elif tag == 50934:
        return 'AsShotProfileName'
    elif tag == 50935:
        return 'NoiseReductionApplied'
    elif tag == 50936:
        return 'ProfileName'
    elif tag == 50937:
        return 'ProfileHueSatMapDims'
    elif tag == 50938:
        return 'ProfileHueSatMapData1'
    elif tag == 50939:
        return 'ProfileHueSatMapData2'
    elif tag == 50940:
        return 'ProfileToneCurve'
    elif tag == 50941:
        return 'ProfileEmbedPolicy'
    elif tag == 50942:
        return 'ProfileCopyright'
    elif tag == 50964:
        return 'ForwardMatrix1'
    elif tag == 50965:
        return 'ForwardMatrix2'
    elif tag == 50966:
        return 'PreviewApplicationName'
    elif tag == 50967:
        return 'PreviewApplicationVersion'
    elif tag == 50968:
        return 'PreviewSettingsName'
    elif tag == 50969:
        return 'PreviewSettingsDigest'
    elif tag == 50970:
        return 'PreviewColorSpace'
    elif tag == 50971:
        return 'PreviewDateTime'
    elif tag == 50972:
        return 'RawImageDigest'
    elif tag == 50973:
        return 'OriginalRawFileDigest'
    elif tag == 50974:
        return 'SubTileBlockSize'
    elif tag == 50975:
        return 'RowInterleaveFactor'
    elif tag == 50981:
        return 'ProfileLookTableDims'
    elif tag == 50982:
        return 'ProfileLookTableData'
    elif tag == 51008:
        return 'OpcodeList1'
    elif tag == 51009:
        return 'OpcodeList2'
    elif tag == 51022:
        return 'OpcodeList3'
    elif tag == 51023:
        return 'FibicsXML'  #
    elif tag == 51041:
        return 'NoiseProfile'
    elif tag == 51043:
        return 'TimeCodes'
    elif tag == 51044:
        return 'FrameRate'
    elif tag == 51058:
        return 'TStop'
    elif tag == 51081:
        return 'ReelName'
    elif tag == 51089:
        return 'OriginalDefaultFinalSize'
    elif tag == 51090:
        return 'OriginalBestQualitySize'
    elif tag == 51091:
        return 'OriginalDefaultCropSize'
    elif tag == 51105:
        return 'CameraLabel'
    elif tag == 51107:
        return 'ProfileHueSatMapEncoding'
    elif tag == 51108:
        return 'ProfileLookTableEncoding'
    elif tag == 51109:
        return 'BaselineExposureOffset'
    elif tag == 51110:
        return 'DefaultBlackRender'
    elif tag == 51111:
        return 'NewRawImageDigest'
    elif tag == 51112:
        return 'RawToPreviewGain'
    elif tag == 51113:
        return 'CacheBlob'
    elif tag == 51114:
        return 'CacheVersion'
    elif tag == 51123:
        return 'MicroManagerMetadata'
    elif tag == 51125:
        return 'DefaultUserCrop'
    elif tag == 51159:
        return 'ZIFmetadata'  # Objective Pathology Services
    elif tag == 51160:
        return 'ZIFannotations'  # Objective Pathology Services
    elif tag == 51177:
        return 'DepthFormat'
    elif tag == 51178:
        return 'DepthNear'
    elif tag == 51179:
        return 'DepthFar'
    elif tag == 51180:
        return 'DepthUnits'
    elif tag == 51181:
        return 'DepthMeasureType'
    elif tag == 51182:
        return 'EnhanceParams'
    elif tag == 52525:
        return 'ProfileGainTableMap'  # DNG 1.6
    elif tag == 52526:
        return 'SemanticName'  # DNG 1.6
    elif tag == 52528:
        return 'SemanticInstanceID'  # DNG 1.6
    elif tag == 52536:
        return 'MaskSubArea'  # DNG 1.6
    elif tag == 52543:
        return 'RGBTables'  # DNG 1.6
    elif tag == 52529:
        return 'CalibrationIlluminant3'  # DNG 1.6
    elif tag == 52531:
        return 'ColorMatrix3'  # DNG 1.6
    elif tag == 52530:
        return 'CameraCalibration3'  # DNG 1.6
    elif tag == 52538:
        return 'ReductionMatrix3'  # DNG 1.6
    elif tag == 52537:
        return 'ProfileHueSatMapData3'  # DNG 1.6
    elif tag == 52532:
        return 'ForwardMatrix3'  # DNG 1.6
    elif tag == 52533:
        return 'IlluminantData1'  # DNG 1.6
    elif tag == 52534:
        return 'IlluminantData2'  # DNG 1.6
    elif tag == 53535:
        return 'IlluminantData3'  # DNG 1.6
    elif tag == 52544:
        return 'ProfileGainTableMap2'  # DNG 1.7
    elif tag == 52547:
        return 'ColumnInterleaveFactor'  # DNG 1.7
    elif tag == 52548:
        return 'ImageSequenceInfo'  # DNG 1.7
    elif tag == 52550:
        return 'ImageStats'  # DNG 1.7
    elif tag == 52551:
        return 'ProfileDynamicRange'  # DNG 1.7
    elif tag == 52552:
        return 'ProfileGroupName'  # DNG 1.7
    elif tag == 52553:
        return 'JXLDistance'  # DNG 1.7
    elif tag == 52554:
        return 'JXLEffort'  # DNG 1.7
    elif tag == 52555:
        return 'JXLDecodeSpeed'  # DNG 1.7
    elif tag == 55000:
        return 'AperioUnknown55000'
    elif tag == 55001:
        return 'AperioMagnification'
    elif tag == 55002:
        return 'AperioMPP'
    elif tag == 55003:
        return 'AperioScanScopeID'
    elif tag == 55004:
        return 'AperioDate'
    elif tag == 59932:
        return 'Padding'
    elif tag == 59933:
        return 'OffsetSchema'
                # Reusable Tags 65000-65535
                # (65000, 'DimapDocumentXML'),
                # EER metadata:
                # (65001, 'AcquisitionMetadata'),
                # (65002, 'FrameMetadata'),
                # (65006, 'ImageMetadata'),
                # (65007, 'PosSkipBits'),
                # (65008, 'HorzSubBits'),
                # (65009, 'VertSubBits'),
                # Photoshop Camera RAW EXIF tags:
                # (65000, 'OwnerName'),
                # (65001, 'SerialNumber'),
                # (65002, 'Lens'),
                # (65024, 'KodakKDCPrivateIFD'),
                # (65100, 'RawFile'),
                # (65101, 'Converter'),
                # (65102, 'WhiteBalance'),
                # (65105, 'Exposure'),
                # (65106, 'Shadows'),
                # (65107, 'Brightness'),
                # (65108, 'Contrast'),
                # (65109, 'Saturation'),
                # (65110, 'Sharpness'),
                # (65111, 'Smoothness'),
                # (65112, 'MoireFilter'),
    elif tag == 65200:
        return 'FlexXML'
    raise KeyError("Unknown tag code")