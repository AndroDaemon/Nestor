# Сторонние компоненты

Код Nestor распространяется под лицензией [MIT](LICENSE). Приложение использует
сторонние компоненты и веса моделей, выпущенные под собственными лицензиями.
Атрибуция и условия приведены ниже.

## Модели

| Компонент | Назначение | Лицензия |
|---|---|---|
| [GigaAM v3](https://github.com/salute-developers/GigaAM) — Salute Devices | распознавание русской речи | MIT |
| [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | сегментация говорящих | MIT |
| [NVIDIA TitaNet (small)](https://huggingface.co/nvidia/speakerverification_en_titanet_large) | голосовые отпечатки | CC-BY-4.0 |

## Библиотеки и рантаймы

| Компонент | Назначение | Лицензия |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | рантайм ASR и диаризации | Apache-2.0 |
| FastAPI, Starlette, Uvicorn, Pydantic | веб-интерфейс | MIT / BSD |
| pywebview | окно приложения | BSD-3-Clause |
| Microsoft Edge WebView2 | движок окна | условия Microsoft |
| PyTorch, Torchaudio | вычисления | BSD-3-Clause |
| FFmpeg | декодирование аудио (внешний процесс, не слинкован) | LGPL / GPL |
| NumPy, SciPy, python-docx, SentencePiece, PyYAML, ONNX Runtime | — | свободные лицензии |

## Веса моделей

Сами веса моделей не входят в git-репозиторий: они либо скачиваются при первом
запуске, либо пакуются в установщик вместе с уведомлениями об атрибуции.
Атрибуция моделей (NVIDIA, pyannote, GigaAM) приведена в этом файле, в
[README.md](README.md) и в разделе «Помощь» приложения.
