"""Interface web : raccourcis clavier C / G / I / Échap et « Non assigné ».

La logique pure de la page vit dans ``src/web/interface.js`` (sans DOM) : elle
est exécutée ici par Node. Le branchement de la page (boutons portant leur
touche, gestionnaire clavier, compteurs) est vérifié sur le HTML et le
JavaScript servis. Sans Node, les tests d'exécution sont ignorés.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "web"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js absent : logique JS non exécutable")


def run_interface(script: str):
    """Exécute ``script`` avec ``I`` = interface.js ; renvoie le JSON imprimé."""
    code = f"const I = require({json.dumps(str(WEB / 'interface.js'))});\n{script}"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


@needs_node
@pytest.mark.parametrize(
    "key, action",
    [
        ("c", "valider"), ("C", "valider"),
        ("g", "recommencer"), ("G", "recommencer"),
        ("i", "inverser"), ("I", "inverser"),
        ("Escape", "retour"), ("Esc", "retour"),
        ("x", None), ("Enter", None), (" ", None),
    ],
)
def test_touches_de_l_ecran_de_tracage(key, action):
    assert run_interface(f"console.log(JSON.stringify(I.actionPourTouche({{key: {json.dumps(key)}}})))") == action


@needs_node
def test_touches_avec_modificateur_ignorees():
    # Ctrl+C (copier), Cmd+G, Alt+I : laissés au navigateur.
    result = run_interface(
        "console.log(JSON.stringify([{key:'c',ctrlKey:true},{key:'g',metaKey:true},{key:'i',altKey:true}]"
        ".map(I.actionPourTouche)))"
    )
    assert result == [None, None, None]


@needs_node
def test_non_assigne_en_mode_sans_ligne():
    result = run_interface(
        "console.log(JSON.stringify([I.texteCompteur(null), I.texteCompteur(undefined), I.texteCompteur(0),"
        " I.texteCompteur(12), I.texteMode(true), I.texteMode(false)]))"
    )
    assert result == [
        {"texte": "Non assigné", "attenue": True},
        {"texte": "Non assigné", "attenue": True},
        {"texte": "0", "attenue": False},
        {"texte": "12", "attenue": False},
        "Sans ligne",
        "Avec ligne",
    ]


@needs_node
def test_aide_affichee_identique_aux_raccourcis():
    """Le panneau d'aide du HTML liste exactement les commandes de interface.js."""
    aide = run_interface("console.log(JSON.stringify(I.AIDE))")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    affichee = re.findall(r"<dt><kbd>([^<]+)</kbd></dt><dd>([^<]+)</dd>", html)
    assert [list(item) for item in affichee] == aide


@needs_node
def test_chaque_bouton_porte_sa_touche():
    """Boutons de l'écran de traçage : la touche affichée déclenche la même action."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    attendu = {"btn-valider": "valider", "btn-recommencer": "recommencer",
               "btn-inverser": "inverser", "btn-retour": "retour"}
    for bouton, action in attendu.items():
        touche = re.search(rf'id="{bouton}" data-touche="([^"]+)"', html).group(1)
        assert run_interface(f"console.log(JSON.stringify(I.actionPourTouche({{key: {json.dumps(touche)}}})))") == action


def test_page_branchee_sur_les_raccourcis_et_les_compteurs():
    """app.js délègue à interface.js : clavier, « Non assigné », mode ; aucun reste du tiret."""
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'addEventListener("keydown"' in app and "Interface.actionPourTouche(ev)" in app
    for action in ("valider", "recommencer", "inverser", "retour"):
        assert f"{action}: {action}" in app
    assert "Interface.texteCompteur(valeur)" in app and 'classList.toggle("non-assigne"' in app
    assert "Interface.texteMode(v.sans_ligne)" in app
    assert "\u2014" not in app
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/interface.js") < html.index("/static/app.js")
    assert ".valeur.non-assigne" in (WEB / "style.css").read_text(encoding="utf-8")


def test_ecran_d_accueil_explique_les_deux_choix():
    html = re.sub(r"\s+", " ", (WEB / "index.html").read_text(encoding="utf-8"))
    assert "Voulez-vous tracer une ligne virtuelle ?" in html
    assert ("Le système compte les entrées et les sorties à travers la ligne, en plus des personnes "
            "présentes. Tracez-la le long du seuil de la porte, pour que toute la salle se trouve du "
            "même côté de la ligne prolongée.") in html
    assert ("Le système compte les personnes présentes au démarrage et celles qui apparaissent "
            "ensuite. Les entrées et les sorties ne sont pas comptées.") in html


@needs_node
def test_etat_relecture_et_etapes_de_demarrage():
    result = run_interface(
        "console.log(JSON.stringify([I.texteEtat('en_cours', false), I.texteEtat('en_cours', true),"
        " I.texteEtat('termine', true), I.texteEtat('hors_ligne', true),"
        " I.texteEtape('chargement_modeles'), I.texteEtape('demarrage')]))"
    )
    assert result == [
        "En cours", "Relecture", "Relecture terminée", "Connexion perdue",
        "Chargement des modèles…", "Démarrage du traitement…",
    ]


def test_calibration_sans_dessin_dans_la_page():
    """La page ne dessine rien : chaque action part au serveur, qui renvoie l'image."""
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "getContext" not in app and "setLineDash" not in app and "fillStyle" not in app
    assert '"/calibration/action"' in app
    for action in ("ligne", "sans_ligne", "clic", "valider", "recommencer", "inverser", "retour"):
        assert f'action: "{action}"' in app
    assert "Interface.texteEtat(valeur, relecture)" in app and "Interface.texteEtape(v.etape)" in app
