# Stable Diffusion Forge sur Mac Pro 5.1 (Ubuntu - No-AVX / No-Cuda - AMD ROCm)

Ce dépôt documente l'installation et l'optimisation de **SD-WebUI-Forge** sur un Mac Pro 5.1 (2010/2012). Ce guide est le fruit d'un travail collaboratif entre un utilisateur passionné et une intelligence artificielle (Gemini), conçu pour repousser les limites de l'architecture Westmere.

---

## ⚠️ Le Défi Matériel : Architecture Westmere & Bus PCIe 2.0
Le processeur Intel Xeon de cette machine est dépourvu des instructions **AVX/AVX-2**. Presque tous les environnements IA modernes les exigent nativement. **Ce guide documente la compilation d'un environnement 100% compatible No-AVX.**

### 🛠 Configuration de test
* **CPU :** Dual Intel Xeon (Westmere) - 2 x 3.33Ghz - **Sans AVX**.
* **RAM :** 128 Go DDR3 ECC (Indispensable pour le swap CPU/GPU). ⚠️ 64 Go DDR3 ECC minimum pour la compilation de Pytorch
* **GPU :** AMD Radeon RX 6600 XT (**8 Go VRAM**).
* **Bus :** PCIe 2.0 (Goulot d'étranglement majeur).
* **System :** Ubuntu 24 LTS

### ⚠️ Installation spécifique et portable `/home/User/IA/`
Contrairement à une installation classique, nous avons fait le choix d'une installation autonome (Sandboxed).
Voici pourquoi cette nature d'installation est vitale pour le Mac Pro 5.1 :

Immunité aux mises à jour système : Ubuntu peut mettre à jour son Python natif ou ses bibliothèques (comme NumPy). Si nous utilisions le Python système, une simple mise à jour 'apt upgrade' pourrait écraser tes binaires No-AVX/No-Cuda par des versions AVX/Cuda standards, rendant l'IA inutilisable instantanément.

Zéro conflit de permissions : En installant tout dans le dossier utilisateur (~/IA/), pas besoin d'utiliser `sudo` pour installer des modules Python. Cela évite de corrompre les permissions du système.

Portabilité : l'environnement IA/ est techniquement "déplaçable". Si on réinstalle le système sur un autre disque, on peux potentiellement pointer vers ce dossier et retrouver ton environnement prêt à l'emploi.

---

## 🛡️ Étape 1 : Validation de ROCm
Avant toute tentative de compilation, il est impératif que **ROCm** soit installé avec ses outils de développement et fonctionnel sur votre système Ubuntu. Il est autement recommandé d'utiliser les versions de AMD.

**1-1. Vérification de l'installation :**
```bash
rocm-smi
```
⚠️ Assurez-vous d'avoir installé les headers nécessaires (ex: rocm-dev, rock-dkms).

**1-2. Permissions**
```bash
sudo usermod -aG video $USER
sudo usermod -aG render $USER
```

---

## ⚙️ 2. Compilation de PyTorch (No-AVX & No-CUDA Hack)

Nous forçons PyTorch à ignorer les instructions AVX et désactivons explicitement NVIDIA CUDA pour une construction ROCm pure.
Il est impératif d'installer ces paquets pour la compilation de Pytorch.
```bash
sudo apt install -y python3-dev python3-pip cmake git build-essential \ libffi-dev libssl-dev libopenblas-dev libblas-dev m4 \ python3-yaml python3-setuptools python3-wheel doxygen \ python3-packaging python3-pkg-resources
```

**Compilation**
Avant de lancer la compilation, définissez ces variables pour brider les instructions incompatibles avec les Xeon Westmere :
```bash
# Désactivation totale de NVIDIA CUDA
export USE_CUDA=0
export USE_CUDNN=0

# Désactivation totale des optimisations AVX (Architecture Westmere)
export USE_AVX=0
export USE_AVX2=0
export USE_AVX512=0

# Désactivation des bibliothèques dépendantes de l'AVX
export USE_FBGEMM=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0

# Cible GPU : AMD ROCm pour architecture gfx1030 (RX 6600 XT)
export PYTORCH_ROCM_ARCH=gfx1030 
```
Lancement de la compilation

⚠️ Pour respecter la nature isolée de notre installation, nous n'utilisons pas le python système, mais le binaire situé dans notre dossier IA dédié.
```bash
~/IA/Python/bin/python3 setup.py install
```
⚠️ Notes
* La compilation sature les cœurs à 100%. Prévoyez 1h à 4h de calcul (1h30 avec ma configuration).
* La pression RAM est montée à 35% de mes 128Go.
* À certains moments, la compilation ne semble plus rien faire (plus de défilement dans le terminal). Le compilateur est simplement en train de traiter une grosse masse de données.

---

## 🏗️ 3 Le Linkage : La "méthode UNIX" (Build Wheel & Copie Physique)

Une fois la compilation terminée, les fichiers ne sont pas encore prêts à l'usage. Nous utilisons une procédure en deux temps : la création d'un paquet Wheel (étape longue) suivie d'une copie physique UNIX pour verrouiller l'installation dans `~/IA/Python`.

Pourquoi cette procédure est-elle longue ?
Le "Wheel" est une compilation finale qui emballe tous les binaires `.so`, les scripts et les métadonnées dans un seul archive certifiée. Cela garantit que chaque composant est correctement lié au binaire Python de ton dossier 'IA'.

* La copie physique assure que chaque bit est présent là où Python le cherche, sans dépendre de liens symboliques instables sur le bus PCIe 2.0.

* Le verrouillage par les métadonnées empêche Forge de tenter une mise à jour qui écraserait notre travail.

**📂 Les dossiers cibles de la copie physique**
Il y a trois types d'éléments à copier (les fichiers .so compilés spécifiquement pour No-AVX) :

    1. `torch/` : Le cœur du réacteur (binaires C++ et interfaces Python).

    2. `torchvision/` : Le module de traitement d'image.

    3. `*.dist-info` ou `*.egg-info` : (Crucial) La "carte d'identité" du paquet qui indique à Python que PyTorch est bien installé.

**📄 Les fichiers critiques (les bibliothèques partagées)**
À l'intérieur de ces dossiers, ce sont surtout les fichiers avec l'extension .so (Shared Objects) qui sont les plus importants. Ce sont eux qui ont été "sculptés" pour tes Xeon Westmere :

    * libtorch_cpu.so : Version sans instructions AVX.

    * libtorch_hip.so : La passerelle vers ta carte AMD RX 6600 XT.

    * _C.cpython-312-x86_64-linux-gnu.so : Le lien direct entre Python et le code C++.

**🛠 La procédure UNIX exacte (Wheel + UNIX)**

Cette étape est le moment de vérité : on fabrique l'installeur, puis on injecte ses composants manuellement pour un contrôle total.

```bash
# --- A. CRÉATION DU WHEEL (L'étape longue) ---
# On emballe la compilation dans un paquet installable
cd ~/pytorch
~/IA/Python/bin/python3 setup.py bdist_wheel

# --- B. NETTOYAGE DE L'ENVIRONNEMENT ISOLÉ ---
# On prépare le terrain dans IA/Python
rm -rf ~/IA/Python/lib/python3.12/site-packages/torch
rm -rf ~/IA/Python/lib/python3.12/site-packages/torchvision
rm -rf ~/IA/Python/lib/python3.12/site-packages/torch-*.dist-info

# --- C. COPIE PHYSIQUE UNIX (Le Linkage) ---
# 1. On copie le code compilé depuis le build
cp -R ~/pytorch/build/lib.linux-x86_64-cpython-312/torch ~/IA/Python/lib/python3.12/site-packages/
cp -R ~/vision/build/lib.linux-x86_64-cpython-312/torchvision ~/IA/Python/lib/python3.12/site-packages/

# 2. On injecte les métadonnées générées par le Wheel (Le Passeport)
# Indispensable pour que 'pip list' et Forge reconnaissent la version
cp -R ~/pytorch/build/lib.linux-x86_64-cpython-312/torch-*.dist-info ~/IA/Python/lib/python3.12/site-packages/
```
**⚠️ Pourquoi copier les .dist-info issus du Wheel ?**
C'est la sécurité anti-écrasement. Sans ces dossiers, Forge croira que PyTorch est absent. Il lancera alors un `pip install torch`, ce qui téléchargera la version officielle (avec AVX) et détruira instantanément ta compilation. En copiant ces dossiers, on dit au système : "Le PyTorch No-AVX/No-Cuda certifié est déjà là, ne touche à rien".

## ✅ 4. Contrôle de l'Accélération Matérielle et Benchmark Comparatif : CPU vs GPU (ROCm)
Pour valider l'intérêt de tout ce travail de compilation, nous avons mis en place un benchmark comparatif. L'objectif est de prouver par les chiffres que la RX 6600 XT, même bridée par le bus PCIe 2.0 du Mac Pro, pulvérise les Xeon dès qu'il s'agit de calcul matriciel (IA).
Ce test permet aussi de confirmer que le "Linkage" est réussi : si le GPU n'est pas détecté, PyTorch basculerait sur le CPU par défaut, ce qui rendrait l'utilisation de Forge insupportable.

**Procédure**

1. Créer le script de vérification :
```bash
nano ~/IA/Python/check_gpu.py
```
2. Le script
```python
import torch
import time

print(f"--- Rapport de Santé PyTorch ---")
print(f"Version de Torch : {torch.__version__}")
print(f"ROCm / CUDA disponible : {torch.cuda.is_available()}")

def benchmark(device, name):
    # Création de deux matrices massives (4000x4000)
    size = 4000
    # On force le type de donnée en float32 pour le test
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    
    # Warm-up (essentiel pour charger les kernels ROCm en VRAM)
    torch.matmul(a, b)
    
    # Mesure du temps sur 10 itérations
    start = time.time()
    for _ in range(10):
        torch.matmul(a, b)
    end = time.time()
    
    avg_time = (end - start) / 10
    print(f"🚀 [{name}] Temps moyen : {avg_time:.4f} secondes")
    return avg_time

print(f"\n--- Lancement du Benchmark (Mac Pro 5.1) ---")

# Test sur CPU (Xeon Westmere - No AVX)
t_cpu = benchmark("cpu", "Dual Xeon (CPU)")

# Test sur GPU (RX 6600 XT - ROCm)
if torch.cuda.is_available():
    t_gpu = benchmark("cuda", "RX 6600 XT (GPU)")
    acceleration = t_cpu / t_gpu
    print(f"\n✅ VERDICT : Le GPU est {acceleration:.1f}x plus rapide que tes Xeon !")
else:
    print("\n❌ ERREUR : GPU non détecté. Vérifiez le Linkage UNIX et la variable HSA_OVERRIDE_GFX_VERSION.")
```
3.Lancement du test
```bash
~/IA/Python/bin/python3 check_gpu.py
```

---

## 🏗 5. Installation de Forge : Clonage et Méthode "Anti-CUDA"

Cette étape consiste à récupérer les sources de Forge, puis à préparer son environnement de manière chirurgicale pour éviter tout conflit avec NVIDIA CUDA.

**5-1. Clonage du dépôt**
Nous installons Forge dans ton dossier dédié `~/IA/`.
```bash
cd ~/IA
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge forge
cd forge
```
**🛡️ 5-2. Lever le verrou Ubuntu (PEP 668)**
Avant de pouvoir installer les dépendances dans notre dossier `/IA/`, nous devons lever la protection d'Ubuntu qui bloque l'usage de pip.
```bash
mkdir -p ~/.config/pip
echo -e "[global]\nbreak-system-packages = true" > ~/.config/pip/pip.conf
```
**🛠️ 5-3. Procédure d'installation chirurgicale**
**!!! ATTENTION : NE LANCEZ JAMAIS 'python3 launch.py' Àà CE STADE !!!**
L'installeur automatique écraserait ton PyTorch "No-AVX". Nous installons les composants un par un manuellement.
```bash
# Définition du chemin vers notre binaire sécurisé
BIN_PY=~/IA/Python/bin/python3

# A. NumPy (Le verrou de version) : impératif en v1.26.4
$BIN_PY -m pip install numpy==1.26.4

# B. Dépendances de l'interface et du backend
$BIN_PY -m pip install gradio==3.48.0 safetensors gguf --upgrade

# C. Modules Vision, CLIP et post-traitement
$BIN_PY -m pip install clip open_clip_torch facexlib gfpgan
```
**Pourquoi cloner AVANT d'installer les modules ?**
Techniquement, on pourrait installer les modules n'importe quand, mais en clonant d'abord, on s'assure d'être dans le bon répertoire de travail. De plus, cela permet de vérifier si Forge n'a pas un fichier requirements.txt spécifique que l'on pourrait consulter en cas de doute.

---

## 🚀 6. Le Script de Lancement : `start_forge.sh`

C'est l'étape finale. Pour que Forge utilise ton environnement isolé ~/IA/Python et ta carte RX 6600 XT sans erreur, nous créons un script de lancement dédié. Ce script force l'identité du GPU et empêche Forge de tenter de "réparer" ou de mettre à jour les modules que nous avons si durement compilés.

**6-1. Création du script**
Placez-vous à la racine de votre dossier Forge :
```bash
cd ~/IA/forge
nano start_forge.sh
```

**6-2. Contenu du script (Optimisé Mac Pro 5.1)**
Copiez et collez le code suivant :
```bash
#!/bin/bash 

# 1. Définition du Python isolé (No-AVX)
PYTHON_IA="$HOME/IA/Python/bin/python3"

# 2. Forcer l'architecture RDNA 2 (Navi 23) pour la RX 6600 XT
export HSA_OVERRIDE_GFX_VERSION=10.3.0 

# 3. Nettoyage préventif des processus fantômes pour libérer la VRAM
killall -9 python3 2>/dev/null 

echo "🚀 Lancement de Stable Diffusion Forge (Mode No-AVX)..."

# 4. Lancement avec les flags de sécurité
# --skip-install : INDISPENSABLE pour ne pas écraser votre compilation
# --precision full --no-half : Sécurité pour éviter les erreurs NaN sur Westmere
$PYTHON_IA launch.py \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    --skip-install \
    --precision full \
    --no-half \
    --listen \
    --always-high-vram \
    --attention-pytorch 

# 5. Nettoyage à la fermeture
echo "🧹 Libération de la VRAM AMD..." 
killall -9 python3 2>/dev/null
```

**6-3. Activation et premier lancement**
Il faut maintenant donner les droits d'exécution à ce script pour pouvoir l'utiliser comme une application.
```bash
# Rendre le script exécutable
chmod +x start_forge.sh

# Lancer l'interface
./start_forge.sh
```

**6-4. Accès à l'interface (Navigateur)**
Une fois que le script `./start_forge.sh` affiche la ligne Model loaded in XXXs, Forge est prêt.
Comme nous avons utilisé le flag --listen, l'interface est accessible sur notre réseau local.

Depuis le Mac Pro lui-même : On ouvre un navigateur (Firefox, Brave, Chrome …) et tape l'adresse locale :
`http://0.0.0.0:7860` (ou `http://0.0.0.1:7860` selon ton alias réseau)

Depuis un autre ordinateur de la maison (Mac, PC, Tablette ou Smartphone) :
Il faut utiliser l'adresse IP locale du Mac Pro sur votre réseau.
1. Pour la connaître, tapez hostname -I dans un terminal sur le Mac Pro.
2. Entrez cette adresse dans le navigateur de votre autre appareil en ajoutant le port :7860 à la fin.
Exemple : 'http://192.168.x.xx:7860'

---

## ⛔ MISE EN GARDE : Flux.1, Z-Image Turbo et autres Modèles Lourds
Bien que Forge permette de charger ces modèles, **leur usage est fortement déconseillé** sur cette configuration :
1. **Goulot d'étranglement PCIe 2.0 :** Le transfert des modèles massifs sur un vieux bus présente des risques d'instabilité.
2. **Saturation VRAM :** 8 Go sont insuffisants pour ces modèles, forçant un ralentissement extrême.
3. **Optimisation :** Préférez **SD 1.5** (vitesse) et **SDXL / Pony (Lightning/Turbo)** pour un usage fluide.
