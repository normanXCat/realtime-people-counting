// Lecture seule : reçoit les quatre compteurs par Server-Sent Events.
(function () {
  "use strict";

  var TIRET = "—";
  var etat = document.getElementById("etat");
  var etatTexte = document.getElementById("etat-texte");
  var champs = {
    presentes: document.getElementById("presentes"),
    entrees: document.getElementById("entrees"),
    sorties: document.getElementById("sorties"),
    nouvelles: document.getElementById("nouvelles")
  };
  var textesEtat = { en_cours: "En cours", termine: "Terminé", hors_ligne: "Connexion perdue" };
  var termine = false;

  function afficherEtat(valeur) {
    etat.setAttribute("data-etat", valeur);
    etatTexte.textContent = textesEtat[valeur] || valeur;
  }

  function afficher(nom, valeur) {
    // null = sans objet (mode sans ligne : ni entrée ni sortie).
    champs[nom].textContent = valeur === null || valeur === undefined ? TIRET : String(valeur);
  }

  var source = new EventSource("/stats");
  source.onmessage = function (message) {
    var v = JSON.parse(message.data);
    afficher("presentes", v.presentes);
    afficher("entrees", v.entrees);
    afficher("sorties", v.sorties);
    afficher("nouvelles", v.nouvelles);
    termine = v.etat === "termine";
    afficherEtat(v.etat);
  };
  source.onerror = function () {
    // Après la fin, on garde « Terminé » et les valeurs finales.
    if (!termine) { afficherEtat("hors_ligne"); }
  };
  source.onopen = function () {
    if (!termine) { afficherEtat("en_cours"); }
  };
})();
