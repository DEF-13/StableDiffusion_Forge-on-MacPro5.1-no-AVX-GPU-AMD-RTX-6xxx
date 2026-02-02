🧀 Projet "Cheese Grater" AI : Stable Diffusion Forge sur Mac Pro 5.1
Guide Ultime : Ubuntu 24.04, Python 3.12, No-AVX & AMD RX 6000

Ce guide documente la réussite technique du déploiement de l'IA générative sur une architecture de 2010.
1. Pré-requis & Environnement (La Base)
Matériel

    Machine : Mac Pro 5.1 (Xeon Westmere).

    RAM : 128 Go recommandés (La pression monte à ~35 Go / 28% durant la compilation).

    GPU : AMD Radeon RX 6600 XT (Architecture gfx1032).

    OS : Ubuntu 24.04 LTS.

Logiciels & Outils de Dev (Indispensable)

    Python : 3.12.3 avec le paquet python3-packaging (obligatoire pour la gestion des versions durant la build).

    ROCm : Version 6.0 ou 6.1 (utilisée pour ce projet).

        Attention : Lors de l'installation de ROCm, vous devez inclure les outils de développement (souvent optionnels) : sudo amdgpu-install --usecase=rocm,hiplibsdk,dkms

2. Script de Diagnostic Automatique (À lancer AVANT de compiler)

Copiez ce code dans un fichier check_system.py et lancez-le avec python3 check_system.py. Il validera si votre Mac Pro est prêt pour la suite.
Python

import os
import subprocess
import sys

def check_step(name, condition, fix_msg):
    status = "✅ OK" if condition else "❌ ERREUR"
    print(f"[{status}] {name}")
    if not condition: print(f"   👉 Solution : {fix_msg}")

print("--- DIAGNOSTIC SYSTÈME MAC PRO 5.1 IA ---")

# 1. Check AVX (Doit être absent sur Westmere)
cpu_info = subprocess.check_output("lscpu", shell=True).decode()
has_avx = "avx" in cpu_info.lower()
check_step("Absence d'AVX (Normal pour Westmere)", not has_avx, "Votre CPU semble supporter l'AVX, ce guide reste valide mais la compilation 'No-AVX' n'est pas strictement obligatoire.")

# 2. Check ROCm & Tools
rocm_path = os.path.exists("/opt/rocm")
check_step("Installation ROCm", rocm_path, "Installez ROCm via amdgpu-install.")

# 3. Check Packaging
try:
    import packaging
    pkg_ok = True
except ImportError:
    pkg_ok = False
check_step("Python Packaging", pkg_ok, "Lancez : sudo apt install python3-packaging")

# 4. Check GPU AMD
try:
    gpu_check = subprocess.check_output("rocminfo", shell=True).decode()
    has_gpu = "gfx1032" in gpu_check
except:
    has_gpu = False
check_step("Détection RX 6600 XT (gfx1032)", has_gpu, "Vérifiez vos drivers ROCm et le support de votre carte.")

3. L'Épreuve de la Compilation : PyTorch No-AVX
Préparation et Variables

On force l'exclusion des instructions que le Xeon ne comprend pas :
Bash

export USE_CUDA=0
export USE_AVX=0
export USE_AVX2=0
export USE_FBGEMM=0
export USE_MKLDNN=0
export PYTORCH_ROCM_ARCH=gfx1032
export CFLAGS="-mno-avx -march=native"
export CXXFLAGS="-mno-avx -march=native"

⏳ Ce qu'il va se passer (Le "Journal de Bord")

La compilation est un marathon. Voici ce à quoi vous devez vous attendre :

    Charge CPU : Vos 12 ou 24 cœurs seront sollicités à 100% pendant toute la durée. La machine va chauffer, c'est normal.

    Pression RAM : Sur 128 Go, l'occupation montera jusqu'à environ 35 Go (28%). Si vous avez moins de 32 Go, assurez-vous d'avoir un "Swap" solide.

    Rythme du terminal : * Phases Rapides : Le texte défile à toute vitesse (compilation des petits modules).

        Phases "Gelées" : Le terminal peut ne plus bouger pendant 10 ou 15 minutes (compilation des gros kernels C++). Ne coupez jamais le processus tant qu'il n'y a pas d'erreur explicite.

4. Finalisation : Forge & Le Patch Gradio (Python 3.12)

Une fois PyTorch installé, installez Forge, mais ne lancez pas l'interface tout de suite.
Le Patch Chirurgical (Indispensable pour Python 3.12)

Le serveur ASGI plantera avec une erreur TypeError ou APIInfoParseError si vous ne modifiez pas manuellement le fichier suivant :

Fichier : ~/.local/lib/python3.12/site-packages/gradio_client/utils.py

Dans les fonctions get_type et _json_schema_to_python_type, insérez impérativement cette barrière de sécurité :
Python

if not isinstance(schema, dict):
    return "Any"

C'est cette modification qui permet à l'interface de s'afficher sur votre navigateur.
5. Note de l'auteur & Remerciements

Ce projet a été mené par François, qui a assuré le rôle d'ingénieur système "sur le métal", gérant les sauvegardes (.bak) et les tests de stabilité en temps réel. Il a été assisté par Gemini (AI) pour la stratégie de compilation et le débugging du code source de Gradio.

Conseil final : Gardez toujours une copie de votre utils.py patché dans un dossier de backup. Une mise à jour de pip peut l'écraser, et cette documentation sera votre seule bouée de sauvetage.

François, voilà une documentation blindée. Elle est à la fois technique, préventive et pédagogique. Elle est prête pour le partage !
