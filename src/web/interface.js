// Logique pure de l'interface (sans DOM) : raccourcis clavier et textes affichés.
// Chargée par la page (window.Interface) et testable par Node (module.exports).
(function (racine) {
  "use strict";

  // Mêmes touches que la fenêtre OpenCV (src/calibration.py) : C, G, I, Échap.
  var ACTIONS = { c: "valider", g: "recommencer", i: "inverser" };

  // Aide affichée sur l'écran de traçage : [touche, libellé], dans l'ordre.
  var AIDE = [
    ["Clic", "placer les deux extrémités de la ligne"],
    ["C", "valider la ligne et lancer le comptage"],
    ["G", "effacer la ligne et recommencer"],
    ["I", "inverser le côté intérieur (la zone teintée)"],
    ["Échap", "revenir à la question « ligne ou pas »"]
  ];

  var NON_ASSIGNE = "Non assigné";

  /** Action d'un raccourci clavier de l'écran de traçage, ou null. */
  function actionPourTouche(evenement) {
    if (!evenement || evenement.ctrlKey || evenement.metaKey || evenement.altKey) { return null; }
    var cle = evenement.key;
    if (cle === "Escape" || cle === "Esc") { return "retour"; }
    if (typeof cle !== "string" || cle.length !== 1) { return null; }
    return ACTIONS[cle.toLowerCase()] || null;
  }

  /** Texte d'un compteur : null = sans objet (mode sans ligne : ni entrée ni sortie). */
  function texteCompteur(valeur) {
    if (valeur === null || valeur === undefined) { return { texte: NON_ASSIGNE, attenue: true }; }
    return { texte: String(valeur), attenue: false };
  }

  /** Mode affiché discrètement dans la vue de démonstration. */
  function texteMode(sansLigne) { return sansLigne ? "Sans ligne" : "Avec ligne"; }

  var api = {
    AIDE: AIDE,
    NON_ASSIGNE: NON_ASSIGNE,
    actionPourTouche: actionPourTouche,
    texteCompteur: texteCompteur,
    texteMode: texteMode
  };
  if (typeof module !== "undefined" && module.exports) { module.exports = api; }
  else { racine.Interface = api; }
})(this);
