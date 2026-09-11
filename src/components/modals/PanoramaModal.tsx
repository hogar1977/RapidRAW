import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle, XCircle, Loader2, Save, RefreshCw, Layers } from 'lucide-react';
import { motion } from 'framer-motion';
import ReactCrop, { type PercentCrop, type Crop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

export type PanoramaProjection = 'spherical' | 'cylindrical' | 'perspective';

export interface PanoramaCropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PanoramaDroppedImage {
  filename: string;
  reason: string;
}

interface PanoramaModalProps {
  crop: PanoramaCropRect | null;
  dropped: Array<PanoramaDroppedImage>;
  error: string | null;
  filenames: Array<string>;
  finalImageBase64: string | null;
  imageCount?: number;
  isOpen: boolean;
  isProcessing: boolean;
  loadingImageUrl?: string | null;
  onClose(): void;
  onOpenFile(path: string): void;
  onProjectionChange(projection: PanoramaProjection): void;
  onSave(crop: PanoramaCropRect | null): Promise<string>;
  onStitch(): void;
  overlayBase64: string | null;
  previewHeight: number;
  previewWidth: number;
  progressMessage: string | null;
  recommendedProjection: string | null;
  saveProgressMessage: string | null;
  saveProgressPercent: number | null;
  selectedProjection: string | null;
  winnerMapBase64: string | null;
}

const PROJECTIONS: PanoramaProjection[] = ['spherical', 'cylindrical', 'perspective'];

function cropToPercent(crop: PanoramaCropRect | null): PercentCrop {
  if (!crop) {
    return { unit: '%', x: 0, y: 0, width: 100, height: 100 };
  }
  return {
    unit: '%',
    x: crop.x * 100,
    y: crop.y * 100,
    width: crop.width * 100,
    height: crop.height * 100,
  };
}

function percentToCrop(crop: Crop): PanoramaCropRect {
  const x = (crop.unit === '%' ? crop.x : crop.x) / (crop.unit === '%' ? 100 : 1);
  const y = (crop.unit === '%' ? crop.y : crop.y) / (crop.unit === '%' ? 100 : 1);
  const width = (crop.unit === '%' ? crop.width : crop.width) / (crop.unit === '%' ? 100 : 1);
  const height = (crop.unit === '%' ? crop.height : crop.height) / (crop.unit === '%' ? 100 : 1);
  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
    width: Math.max(0.01, Math.min(1 - x, width)),
    height: Math.max(0.01, Math.min(1 - y, height)),
  };
}

export default function PanoramaModal({
  crop,
  dropped,
  error,
  filenames,
  finalImageBase64,
  imageCount,
  isOpen,
  isProcessing,
  loadingImageUrl,
  onClose,
  onOpenFile,
  onProjectionChange,
  onSave,
  onStitch,
  overlayBase64,
  previewHeight,
  previewWidth,
  progressMessage,
  recommendedProjection,
  saveProgressMessage,
  saveProgressPercent,
  selectedProjection,
  winnerMapBase64,
}: PanoramaModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [localCrop, setLocalCrop] = useState<PercentCrop>(cropToPercent(crop));
  const [hoveredFile, setHoveredFile] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const autoStartedRef = useRef(false);
  const mouseDownTarget = useRef<EventTarget | null>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const winnerImgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    }
    setShow(false);
    const timer = setTimeout(() => {
      setIsMounted(false);
      setSavedPath(null);
      setIsSaving(false);
      autoStartedRef.current = false;
      setHoveredFile(null);
    }, 300);
    return () => clearTimeout(timer);
  }, [isOpen]);

  useEffect(() => {
    setLocalCrop(cropToPercent(crop));
  }, [crop]);

  useEffect(() => {
    if (
      isOpen &&
      !isProcessing &&
      !finalImageBase64 &&
      !error &&
      !savedPath &&
      (imageCount ?? 0) >= 2 &&
      !autoStartedRef.current
    ) {
      autoStartedRef.current = true;
      onStitch();
    }
  }, [isOpen, isProcessing, finalImageBase64, error, savedPath, imageCount, onStitch]);

  useEffect(() => {
    if (!winnerMapBase64) {
      winnerImgRef.current = null;
      return;
    }
    const img = new Image();
    img.onload = () => {
      winnerImgRef.current = img;
    };
    img.src = winnerMapBase64.startsWith('data:')
      ? winnerMapBase64
      : `data:image/png;base64,${winnerMapBase64}`;
  }, [winnerMapBase64]);

  const handleClose = useCallback(() => {
    if (isSaving) return;
    onClose();
  }, [onClose, isSaving]);

  const handleBackdropMouseDown = (e: React.MouseEvent) => {
    mouseDownTarget.current = e.target;
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && mouseDownTarget.current === e.currentTarget) {
      handleClose();
    }
    mouseDownTarget.current = null;
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const path = await onSave(percentToCrop(localCrop));
      setSavedPath(path);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSaving(false);
    }
  };

  const handleOpen = () => {
    if (savedPath) {
      onOpenFile(savedPath);
      handleClose();
    }
  };

  const handlePointer = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!previewRef.current || !winnerImgRef.current || !previewWidth || !previewHeight) {
      setHoveredFile(null);
      return;
    }
    const rect = previewRef.current.getBoundingClientRect();
    const containerAspect = rect.width / rect.height;
    const imageAspect = previewWidth / previewHeight;
    let drawW = rect.width;
    let drawH = rect.height;
    let offsetX = 0;
    let offsetY = 0;
    if (containerAspect > imageAspect) {
      drawW = rect.height * imageAspect;
      offsetX = (rect.width - drawW) / 2;
    } else {
      drawH = rect.width / imageAspect;
      offsetY = (rect.height - drawH) / 2;
    }
    const lx = e.clientX - rect.left - offsetX;
    const ly = e.clientY - rect.top - offsetY;
    if (lx < 0 || ly < 0 || lx > drawW || ly > drawH) {
      setHoveredFile(null);
      return;
    }
    const gx = Math.floor((lx / drawW) * winnerImgRef.current.width);
    const gy = Math.floor((ly / drawH) * winnerImgRef.current.height);
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(winnerImgRef.current, gx, gy, 1, 1, 0, 0, 1, 1);
    const pixel = ctx.getImageData(0, 0, 1, 1).data[0];
    if (pixel === 0) {
      setHoveredFile(null);
      return;
    }
    const idx = pixel - 1;
    if (idx >= 0 && idx < filenames.length) {
      setHoveredFile(filenames[idx]);
      setTooltipPos({ x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 20 });
    } else {
      setHoveredFile(null);
    }
  };

  const renderContent = () => {
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center py-10 h-[460px]">
          <div className="flex items-center justify-center mb-6">
            <XCircle className="w-12 h-12 text-red-500" />
          </div>
          <Text variant={TextVariants.title} className="mb-2 text-center">
            {t('modals.panorama.failed')}
          </Text>
          <Text className="text-left p-4 rounded-lg bg-bg-primary max-w-2xl mt-2 leading-relaxed whitespace-pre-wrap font-mono text-xs max-h-[320px] overflow-y-auto">
            {String(error)}
          </Text>
        </div>
      );
    }

    if (finalImageBase64 && !isProcessing) {
      return (
        <div className="w-full flex flex-col md:flex-row gap-4">
          <div
            ref={previewRef}
            onPointerMove={handlePointer}
            onPointerLeave={() => setHoveredFile(null)}
            className="relative flex-1 max-h-[500px] bg-[#111] rounded-lg overflow-hidden border border-surface flex items-center justify-center"
          >
            <ReactCrop crop={localCrop} onChange={(_, percent) => setLocalCrop(percent)} keepSelection>
              <div className="relative inline-block max-h-[500px]">
                <img
                  src={finalImageBase64}
                  alt="Stitched Panorama"
                  className="max-w-full max-h-[500px] object-contain"
                />
                {overlayBase64 && (
                  <img
                    src={overlayBase64.startsWith('data:') ? overlayBase64 : `data:image/png;base64,${overlayBase64}`}
                    alt=""
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none opacity-80"
                  />
                )}
              </div>
            </ReactCrop>
            {hoveredFile && (
              <div
                className="absolute z-20 bg-black/80 text-white text-xs font-mono px-2 py-1 rounded pointer-events-none"
                style={{ left: tooltipPos.x, top: tooltipPos.y }}
              >
                {hoveredFile}
              </div>
            )}
          </div>
          <div className="w-full md:w-56 shrink-0 space-y-4">
            <div>
              <Text variant={TextVariants.small} className="uppercase tracking-wide opacity-60 mb-2">
                {t('modals.panorama.projection')}
              </Text>
              <div className="space-y-2">
                {PROJECTIONS.map((p) => (
                  <label key={p} className="flex items-center gap-2 text-sm cursor-pointer">
                    <input
                      type="radio"
                      name="projection"
                      checked={(selectedProjection || recommendedProjection) === p}
                      onChange={() => onProjectionChange(p)}
                    />
                    <span className="capitalize">
                      {t(`modals.panorama.projections.${p}`)}
                      {recommendedProjection === p ? ` (${t('modals.panorama.recommended')})` : ''}
                    </span>
                  </label>
                ))}
              </div>
            </div>
            {dropped.length > 0 && (
              <div className="p-2 rounded-lg border border-amber-700/40 bg-amber-950/30 text-xs text-amber-200">
                <div className="font-semibold mb-1">
                  {t('modals.panorama.excluded', { count: dropped.length })}
                </div>
                <ul className="space-y-1 list-disc list-inside">
                  {dropped.map((d, i) => (
                    <li key={i} title={d.reason} className="truncate font-mono">
                      {d.filename}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {savedPath && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
                <Text
                  as="div"
                  variant={TextVariants.heading}
                  color={TextColors.success}
                  className="flex items-center gap-2"
                >
                  <CheckCircle className="w-5 h-5" />
                  <span>{t('modals.panorama.savedSuccess')}</span>
                </Text>
              </motion.div>
            )}
          </div>
        </div>
      );
    }

    if (isProcessing) {
      return (
        <div className="flex h-[460px] overflow-hidden rounded-lg border border-surface">
          <div className="w-2/5 relative overflow-hidden shrink-0 bg-[#0a0a0a] flex items-center justify-center">
            {loadingImageUrl ? (
              <img src={loadingImageUrl} alt="Source preview" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full bg-surface/50" />
            )}
          </div>
          <div className="flex-1 flex flex-col items-center justify-center px-12 bg-bg-primary">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.4 }}
              className="flex flex-col items-center w-full"
            >
              <Text variant={TextVariants.title} className="mb-2 text-center">
                {t('modals.panorama.stitchingProgress')}
              </Text>
              <Text className="text-center font-mono h-6 flex justify-center items-center">
                {progressMessage || t('modals.panorama.initializing')}
              </Text>
              <div className="mt-8 w-64 relative">
                <div className="h-1 bg-surface rounded-full overflow-hidden relative w-full shadow-xs">
                  <motion.div
                    className="absolute inset-y-0 w-[80%] bg-linear-to-r from-transparent via-accent to-transparent mix-blend-screen"
                    style={{ filter: 'blur(3px)' }}
                    animate={{ x: ['-150%', '150%'] }}
                    transition={{ repeat: Infinity, duration: 1.5, ease: [0.4, 0, 0.2, 1] }}
                  />
                </div>
              </div>
              <Text variant={TextVariants.small} className="mt-6 text-center max-w-xs opacity-60">
                {t('modals.panorama.speedNotice')}
              </Text>
            </motion.div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col items-center justify-center h-[460px]">
        <div className="flex items-center justify-center mb-6">
          <Layers className="w-12 h-12 text-accent" />
        </div>
        <Text variant={TextVariants.title} className="mb-3 text-center">
          {t('modals.panorama.title')}
        </Text>
        <Text className="text-center max-w-md leading-relaxed text-text-secondary">
          {imageCount ? t('modals.panorama.descCount', { count: imageCount }) : t('modals.panorama.descGeneric')}
        </Text>
      </div>
    );
  };

  const renderButtons = () => {
    if (error) {
      return (
        <Button onClick={handleClose} className="w-full">
          {t('modals.panorama.close')}
        </Button>
      );
    }

    if (savedPath) {
      return (
        <>
          <button
            onClick={handleClose}
            className="px-4 py-2 rounded-md text-text-secondary hover:bg-card-active transition-colors"
          >
            {t('modals.panorama.close')}
          </button>
          <Button onClick={handleOpen}>{t('modals.panorama.openInEditor')}</Button>
        </>
      );
    }

    const showSaveProgress = isSaving || saveProgressPercent != null;
    const percent = saveProgressPercent ?? 0;

    return (
      <div className="w-full flex items-center gap-3">
        {showSaveProgress && (
          <div className="flex-1 min-w-0 flex flex-col justify-center gap-1.5 pr-1">
            <div className="flex items-center justify-between gap-3 text-xs text-text-secondary">
              <span className="truncate font-mono">
                {saveProgressMessage || t('modals.panorama.savingProgress', { percent: Math.round(percent) })}
              </span>
              <span className="shrink-0 tabular-nums">{Math.round(percent)}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-surface overflow-hidden">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-200 ease-out"
                style={{ width: `${Math.max(2, Math.min(100, percent))}%` }}
              />
            </div>
          </div>
        )}

        <div
          className={`flex items-center justify-end gap-2 shrink-0 ${
            isProcessing || isSaving ? 'opacity-50 pointer-events-none' : ''
          }`}
        >
          <button
            onClick={handleClose}
            className="px-4 py-2 rounded-md text-text-secondary hover:bg-card-active transition-colors text-sm"
            disabled={isSaving}
          >
            {finalImageBase64 ? t('modals.panorama.close') : t('modals.panorama.cancel')}
          </button>

          {finalImageBase64 && (
            <Button onClick={onStitch} disabled={isProcessing || isSaving} variant="secondary">
              {isProcessing ? <Loader2 className="animate-spin mr-2" size={16} /> : <RefreshCw className="mr-2" size={16} />}
              {t('modals.panorama.retry')}
            </Button>
          )}

          {finalImageBase64 && (
            <Button onClick={handleSave} disabled={isSaving || isProcessing}>
              {isSaving ? <Loader2 className="animate-spin mr-2" size={16} /> : <Save className="mr-2" size={16} />}
              {t('modals.panorama.save')}
            </Button>
          )}
        </div>
      </div>
    );
  };

  if (!isMounted) return null;

  return (
    <div
      className={`fixed inset-0 flex items-center justify-center z-50 bg-black/40 backdrop-blur-xs transition-opacity duration-300 ease-in-out ${
        show ? 'opacity-100' : 'opacity-0'
      }`}
      onMouseDown={handleBackdropMouseDown}
      onClick={handleBackdropClick}
    >
      <div
        className={`bg-surface rounded-xl shadow-2xl p-6 w-full max-w-5xl transform transition-all duration-300 ease-out ${
          show ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 -translate-y-4'
        }`}
        onClick={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex flex-col">
          {renderContent()}
          <div className={`mt-4 flex w-full ${savedPath ? 'justify-end gap-3' : 'pt-4 border-t border-surface/50'}`}>
            {renderButtons()}
          </div>
        </div>
      </div>
    </div>
  );
}
