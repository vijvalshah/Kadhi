<!-- synced-from: README.md sha256:b8ed65c4ce9df6f6072bbf35f509a22938ed59b37d4a4c101fcfa7dd99a0eb0d -->
<p align="center">🌍 <a href="README.md">English</a> | <strong>Türkçe</strong></p>

<p align="center">
  <img src="kadhi_logo_svg.svg" alt="Kadhi" width="140">
</p>

<h1 align="center">Kadhi</h1>

<p align="center">
  <strong>Tek bir YAML dosyası girin, ince ayarlı bir model çıksın. SSH oturumlarını ve yapılandırma labirentini atlayın.</strong>
</p>

<p align="center">
  <a href="#hızlı-başlangıç">Hızlı Başlangıç</a> &middot;
  <a href="#web-arayüzü">Web Arayüzü</a> &middot;
  <a href="#yapılandırma">Yapılandırma</a> &middot;
  <a href="#belgeler">Belgeler</a> &middot;
  <a href="docs/commands.md">Komutlar</a> &middot;
  <a href="docs/models.md">Modeller</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10--3.12-6D5CE0" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-6D5CE0" alt="Apache-2.0 Lisansı">
</p>

---

İnce ayar bir platform ekibi gerektirmemeli. Kadhi tüm iş akışını — veri, tarif, donanım
algılama, eğitim döngüsü — tek bir YAML dosyasına ve tek bir komuta indirger.

```bash
pip install "kadhi-cli[train]"   # ince ayar için [train] ekleyin; yalın `kadhi-cli` hafif CLI'dır
kadhi init --template chat
kadhi train
```

**4 GB dizüstü GPU'da 8B bir modele ince ayar yapın.** Katman akışı, dondurulmuş tabanı VRAM'in
dışında tutar ve GPU'ya her seferinde bir kod çözücü katman besler. RTX 3050 Laptop 4 GB üzerinde
ölçüldü: Llama-3.1-8B-Instruct + NF4 ile **119.6 tok/s, 3.32 GB tepe** — normal, belleğe tamamen
yerleşik bir çalıştırmayla bit düzeyinde özdeş ve bir H100 üzerinde aynı 3.32 GB'da 113.00 tok/s
ile bağımsız olarak yeniden üretildi. (tok/s rakamı, 32B'de −%4.8'e mal olan v0.73.0 doğruluk
onarımından önce, v0.72.2'de ölçüldü; o zamandan beri 4 GB bir kartta yeniden çalıştırılmadı.)
İsteğe bağlıdır (`stream_layers: true`) ve hâlâ BETA —
[nasıl çalışır](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[tüm ölçümler](benchmarks/) ·
**[ücretsiz bir Colab T4'te kendiniz doğrulayın](notebooks/proof-4gb.ipynb)** (işlemi 4 GB ile
sınırlar, ardından akışlı bir modelin normal bir modelle bit düzeyinde özdeş olduğunu doğrular)

## Kadhi'nın çözdüğü sorun

Deneyimli ML ekipleri bile zamanlarının %30-50'sini model kalitesi yerine altyapı işleriyle
geçiriyor — CUDA uyumsuzluklarını kovalamak, SSH oturumlarına göz kulak olmak, her GPU için
toplu iş boyutunu elle ayarlamak. Kadhi bu katmanı üstlenir; geriye düşünmeniz gereken tek şey
tarif kalır.

- 🖥️ **SSH kazı yok.** Kadhi'ya bir makine gösterin; size hata ayıklamanız gereken bozuk bir kabuk vermez.
- 📄 **Tek YAML, tek gerçek kaynağı.** Dağınık betikler yok, hatırlanması gereken gizli CLI bayrakları yok.
- ⚙️ **Donanımı varsayılan olarak tanır.** Toplu iş boyutu, niceleme ve GPU algılama tahmin edilmez, çıkarılır.
- 🔒 **Kendi donanımınızda çalışır.** Kendi GPU'nuzda QLoRA — zorunlu bulut bağımlılığı yok.

> Yalnızca Python **3.10–3.12**. 3.13+ sürümlerinde pip, Kadhi daha hiç çalışmadan yerel
> eklentide çöken, test edilmemiş PyTorch tekerleklerini çözümlüyordu.

## Hızlı Başlangıç

### 1. Kurulum

Kadhi bir komut satırı uygulamasıdır; bu yüzden en temiz kurulum ona kendi ortamını verir ve
`kadhi`'u `PATH`'inize ekler:

```bash
# Hafif çekirdek: CLI + yapılandırma + veri araçları, PyTorch yok
pipx install kadhi-cli
uv tool install kadhi-cli          # aynı fikir, zaten uv kullanıyorsanız

# Eğitim yığınını ekleyin (torch, transformers, peft, trl, datasets, …)
pipx install "kadhi-cli[train]"

# Her şey (train + serve + ui + data) tek seferde
pipx install "kadhi-cli[all]"
```

Zaten bir virtualenv'in, bir Colab not defterinin ya da bir Docker imajının içinde misiniz? Aynı
adlar ve eklerle doğrudan `pip` kullanın:

```bash
pip install kadhi-cli
pip install "kadhi-cli[train]"
pip install "kadhi-cli[all]"
```

Kendi kodunuzdan da `import kadhi_cli` yapmak istiyorsanız `pipx` yerine `pip` kullanın; çünkü
pipx uygulamayı kasıtlı olarak diğer her şeyden yalıtır.

Eklerin tam tablosu (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …)
[`docs/models.md`](docs/models.md#optional-extras) içinde yer alır.

> **`error: externally-managed-environment` mı?** Bu,
> [PEP 668](https://peps.python.org/pep-0668/)'dir; bir Kadhi sorunu değil. Debian 12,
> Ubuntu 23.04 ve sonrası `pip`'in sistem Python'una yazmasını engeller, çünkü bu dosyaları
> `apt` de yönetir. `pipx` ve `uv tool`, Kadhi'a kendi ortamını vererek bunu aşar; yukarıda ilk
> sırada yer almalarının nedeni budur. `python3 -m venv .venv && source .venv/bin/activate`
> ardından düz `pip` de aynı ölçüde işe yarar.

> **Tek tırnak değil, çift tırnak.** `"kadhi-cli[train]"` her kabukta çalışan tek yazımdır —
> `cmd.exe`, PowerShell, bash ve zsh. `'kadhi-cli[train]'` biçimini eski bir eğitimden
> kopyaladıysanız ve pip reddettiyse, nedeni budur:
> [nedeni ve hatanın tam metni](docs/models.md#quoting-the-extra).

`kadhi init`, `kadhi data …` ve diğer veri/inceleme komutları hafif kurulumda çalışır.
İnce ayar (`kadhi train`) `[train]` ekini gerektirir.

### 2. Bir yapılandırma oluşturun

```bash
kadhi init                       # etkileşimli sihirbaz
kadhi init --template chat       # ya da bir şablondan başlayın
```

Şablonlar: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Eğitin, test edin, yayınlayın

```bash
kadhi train --config kadhi.yaml                 # LoRA, niceleme, toplu işleme — hepsi halledilir
kadhi chat  --model ./output                    # modelinizle konuşun
kadhi push  --model ./output --repo you/my-model

kadhi merge  --adapter ./output                              # LoRA'yı tabana birleştirin
kadhi export --model ./output --format gguf --quant q4_k_m   # Ollama / llama.cpp için GGUF
```

Diğer dışa aktarma hedefleri (ONNX, TensorRT, AWQ, GPTQ, BitNet) ve dağıtım seçenekleri
[`docs/serving-and-export.md`](docs/serving-and-export.md) içinde yer alır.

## Web Arayüzü

Tarayıcı mı tercih edersiniz? `kadhi ui`; deneyler, eğitim kurulumu, canlı ölçümler, veri kümesi
keşfi ve modelle sohbet için yerel bir pano sunar.

```bash
pip install "kadhi-cli[ui]"
kadhi ui
# http://127.0.0.1:7860 adresini açar
```

[Web Arayüzü belgeleri](docs/serving-and-export.md#web-ui)

## Yapılandırma

Eksiksiz bir `kadhi.yaml`:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5 kat daha hızlı, pip install "kadhi-cli[fast]"

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
  quantization: 4bit

output: ./output
```

`config/schema.py` her alanın tek doğru kaynağıdır. Gelişmiş veri, eğitim ve PEFT seçenekleri
[Belgeler](#belgeler) altında belgelenmiştir.

> **Bilinmeyen yapılandırma anahtarları v0.75'ten beri reddediliyor.** Hiçbir modelin
> tanımlamadığı bir anahtar — `quantizaton` gibi bir yazım hatası ya da yalnızca daha yeni bir
> Kadhi sürümünde bulunan bir alan — eskiden şemadan geçip göz ardı ediliyordu; yani çalıştırma,
> o ayar hiç uygulanmadan devam ediyordu. v0.74 bunu yükleme anında, muhtemelen kastettiğiniz
> alanla birlikte raporluyordu; **v0.75**'ten itibaren aynı yapılandırma doğrudan reddediliyor;
> bu yüzden anahtarın yoksayılmasına güvenmek yerine onu düzeltin ya da kaldırın. Bkz.
> [Bilinmeyen yapılandırma anahtarları](docs/backends-and-ops.md#unknown-config-keys).

## Belgeler

Tam özellik başvurusu [`docs/`](docs/) içinde. Buradan başlayın:

| Rehber | Kapsam |
|---|---|
| [Eğitim görevleri ve yöntemleri](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, araç çağırma, PRM, ön eğitim, damıtma, sınıflandırma, görü/ses/TTS, unutturma, RAFT/RA-DIT, döngü sağlamlaştırma dedektörleri |
| [PEFT, uzun bağlam ve verimlilik](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, optimizer ve PEFT koleksiyonu, LLaMA Pro, GaLore, YaRN/LongLoRA, paketleme, müfredat, otomatik ayarlama |
| [Performans ve niceleme](docs/performance-and-quantization.md) | QAT, FP8, Niceleme Menüsü (I + II), KV önbelleği, NVFP4, kayıt biçimleri, Cut Cross-Entropy, gradyan denetim noktası, çekirdekler, aktivasyon boşaltma, katman akışı, çoklu GPU / DeepSpeed / FSDP |
| [Veri mühendisliği](docs/data.md) | Biçimler, Axolotl/LF eşdeğeri ardışık düzen, veri araçları, sentetik üretim ve forge, kalite karneleri, iz (trace) araçları, uzak veri kümeleri, karıştırma, tarif DAG'ları |
| [Değerlendirme ve sondalar](docs/evaluation.md) | Değerlendirme tasarımı/kapısı, değerlendirme kapılı eğitim, kıyaslamalar, NLG ölçütleri, kalibrasyon, Elo arenası, teşhis, eğitim sonrası X-ray sondaları, A/B, kayma, ayarlanabilirlik, `kadhi advise` |
| [Sunum ve dışa aktarma](docs/serving-and-export.md) | OpenAI uyumlu sunucu, toplu çıkarım, kıyaslama, birleştirme/dışa aktarma, Anthropic Messages uç noktası, spekülatif kod çözme (kendi taslak modelinizi eğitin + ölçün), dağıtım otomatik pilotu, Web Arayüzü, Agent Forge |
| [Bağdaştırıcılar, kayıt defteri ve yönetişim](docs/adapters-and-governance.md) | Bağdaştırıcı yaşam döngüsü/yönetimi, model kayıt defteri, Kadhi Cans, veri çarkı (`kadhi loop`), bilgi düzenleme, yönlendirme (steering), tedarik zinciri denetimleri (scan/sign/BOM/attest/audit/airgap) |
| [Uyumluluk ve yönetişim için hızlı başlangıç](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` şablonları, köken bilgisi (BOM/attest/repro-receipt), denetim günlüğü, air-gap, model kartı otomatik üretimi (`kadhi card`), CI kapısı (`kadhi ci init`) |
| [Arka uçlar, platform ve operasyon](docs/backends-and-ops.md) | MLX/Unsloth arka uçları, alternatif hub'lar, HF Hub entegrasyonu, otomatik pilot, deney takibi, plan/apply, ortam kilit dosyaları, donanım uygunluğu, tamamlamalar, eklentiler, yardımcı komutlar |
| [Komut başvurusu](docs/commands.md) | Tam `kadhi` komut listesi |
| [Desteklenen modeller ve ekler](docs/models.md) | Önerilen model aileleri, VRAM boyut rehberi, pip ekleri matrisi |

## Veri Biçimleri

Alpaca, ShareGPT, ChatML, tercih çiftleri (DPO / ORPO / SimPO / IPO / KTO), görü, ses, ASR, düz
metin, gömme (embedding), RAFT ve daha fazlası — hepsi JSONL, JSON, CSV, Parquet veya TXT'den
otomatik algılanır; bu yüzden çoğu durumda `data.train`'i bir dosyaya yöneltirsiniz ve başka hiçbir
şey değişmez. Her biçim için çalışılmış bir örnekle şemalar ve veri ardışık düzeni (uzak URI'ler,
akış, parçalama, iç içe geçirme, sözcük dağarcığı genişletme, belge alımı)
[`docs/data.md`](docs/data.md#data-formats) içinde.

## Yaygın Komutlar

```bash
kadhi train  --config kadhi.yaml        # eğitin (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
kadhi infer  --model ./output --input prompts.jsonl   # toplu çıkarım
kadhi chat   --model ./output          # etkileşimli sohbet
kadhi serve  --model ./output          # OpenAI uyumlu API sunucusu
kadhi ui                               # yerel tarayıcı panosu
kadhi merge  --adapter ./output        # LoRA'yı taban modele birleştirin
kadhi export --model ./output --format gguf           # dağıtım için dışa aktarın
kadhi eval   benchmark --model ./output               # değerlendirin
kadhi data   inspect ./data/train.jsonl               # veri kümesi istatistikleri
kadhi recipes list                     # 100+ hazır model tarifi
kadhi autopilot --model <id> --data d.jsonl --goal chat  # sıfır yapılandırma
kadhi doctor                           # GPU / bağımlılıklar / ortamı denetleyin
```

Tam komut listesi [`docs/commands.md`](docs/commands.md) içinde.

## Desteklenen Modeller

Kadhi, [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) üzerindeki
**herhangi** bir metin üretim modeliyle çalışır — `AutoModelForCausalLM` ile yükleniyorsa, sıfır
yapılandırma değişikliğiyle çalışır. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral, Mixtral, DeepSeek
R1/V3, Phi-4 ve 100'den fazlası hazır tarifler olarak gelir (`kadhi recipes list`).

| VRAM | En büyük model (QLoRA 4-bit) | Örnek |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (tam) veya MoE | Mixtral-8x22B, DeepSeek-V3 |

Tam model + görü tabloları ve isteğe bağlı ekler matrisi [`docs/models.md`](docs/models.md) içinde.

## Docker

CUDA ya da PyTorch'u yerel olarak kurmadan Kadhi'u çalıştırın (imaj her sürümde GHCR'a yayımlanır):

```bash
docker pull ghcr.io/makazhanalpamys/kadhi:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/kadhi train --config kadhi.yaml
docker compose up   # ya da yerel olarak derleyin
```

## Gereksinimler

- Python 3.10, 3.11 veya 3.12 (CI'ın test ettiği sürümler bunlar; PyTorch yığını orada henüz
  doğrulanmadığı için 3.13+ şimdilik desteklenmiyor)
- CUDA'lı GPU (önerilen), Apple Silicon (MPS) veya CPU (deneysel — çok yavaş)
- QLoRA ile 7B modeller için 8 GB+ VRAM

Tüm eğitim görevleri test amacıyla CPU'da çalışır (niceleme otomatik olarak kapatılır). İsteğe bağlı
ekler (`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) [`docs/models.md`](docs/models.md#optional-extras) içinde
listelenmiştir.

## Sorun Giderme

```bash
kadhi doctor    # GPU, sistem kaynakları, bağımlılıklar ve sürüm tek yerde
```

CUDA tekerlekleri, sürüm uyumsuzlukları: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Geliştirme

```bash
git clone <this-repo>
cd kadhi
pip install -e ".[dev]"

ruff check src/kadhi_cli/ tests/    # lint
pytest tests/ -v                   # birim testleri (hızlı, GPU yok)
pytest tests/ -m smoke -v          # duman testleri (küçük bir model indirir, eğitir)

pre-commit install                 # isteğe bağlı: commit'te ruff lint+format
```

Tam iş akışı için [CONTRIBUTING.md](CONTRIBUTING.md)'ye, bir güvenlik açığı bildirmek için
[SECURITY.md](SECURITY.md)'ye bakın. Telemetri kesinlikle isteğe bağlıdır (`KADHI_TELEMETRY=1`,
varsayılan kapalı; bkz. [Gizlilik Politikası](docs/backends-and-ops.md#privacy-policy)).

## Kadhi'u Destekleyin

Kadhi Apache-2.0 lisanslı ve ücretsizdir — ve öyle kalacak. Depoya yıldız vermek hiçbir maliyeti
yoktur ve başkalarının onu bulmasına yardımcı olur.

## Katkıda Bulunanlar

Topluluk tarafından inşa edildi ❤️ — katkıda bulunan herkese teşekkürler. Bkz.
[CONTRIBUTORS.md](CONTRIBUTORS.md).

## İletişim

Hatalar ve özellik istekleri sorun takipçisine, sorular Discussions'a aittir — ikisi de daha
hızlı yanıtlanır ve aynı sorunu yaşayan bir sonraki kişiye yardım eder.
[Davranış Kuralları](CODE_OF_CONDUCT.md) orada da geçerlidir.

Herkese açık olmaya uygun olmayan her şey için — güvenlik bildirimleri, bkz. [SECURITY.md](SECURITY.md).

Bu README'deki performans rakamlarının arkasındaki ölçüm kayıtları, yazıldığı haliyle
[`benchmarks/`](benchmarks/) içinde yayımlanmıştır — başarısızlıklar, yanlış çıkan varsayımlar ve
ölçülüp sonra atılan sayılar dahil.

## Lisans

[Apache-2.0](LICENSE). Telif hakkı © Kadhi katkıda bulunanları.
