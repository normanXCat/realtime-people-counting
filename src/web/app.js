// Interface web : calibration dans le navigateur, puis vue de démonstration.
// Aucune logique de comptage ni aucun dessin ici : la page envoie clics et
// touches au serveur, qui applique la calibration OpenCV et renvoie les images,
// puis elle affiche l'état publié (traitement en direct ou relecture).
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var Interface = window.Interface;

  var vues = { attente: $("vue-attente"), calibration: $("vue-calibration"), demo: $("vue-demo") };
  var vueCourante = null;

  function montrer(nom) {
    if (vueCourante === nom) { return; }
    vueCourante = nom;
    Object.keys(vues).forEach(function (cle) { vues[cle].hidden = cle !== nom; });
    if (nom === "demo") { demarrerDemo(); }
    if (nom === "calibration") { demarrerCalibration(); }
  }

  // ---------------------------------------------------------------------------
  // Calibration : la page ne dessine rien. Chaque clic et chaque touche est
  // envoyé au serveur, qui l'applique à la sélection de la fenêtre OpenCV et
  // renvoie l'image redessinée par les fonctions OpenCV (src/calibration.py).
  // ---------------------------------------------------------------------------
  var calib = { info: null, points: 0, enCours: false, prete: false };
  var image = $("image-calibration");
  var toile = $("trace");
  var message = $("message");

  function dire(texte, erreur) {
    message.textContent = texte || "";
    message.classList.toggle("erreur", !!erreur);
  }

  function rafraichirImage(version) {
    image.src = "/calibration/image?v=" + encodeURIComponent(version);
  }

  function demarrerCalibration() {
    if (calib.prete) { return; }
    calib.prete = true;
    var info = calib.info || {};
    // Page rechargée pendant le tracé : reprendre l'état du serveur.
    calib.points = info.points || 0;
    $("choix-mode").hidden = info.question === false;
    $("outils-ligne").hidden = info.question !== false;
    majOutils();
    rafraichirImage(info.version !== undefined ? info.version : Date.now());
  }

  function majOutils() {
    var n = calib.points;
    $("btn-inverser").disabled = n < 2;
    $("btn-valider").disabled = n < 2 || calib.enCours;
    $("consigne").textContent = n === 0 ? "Cliquez le premier point de la ligne."
      : n === 1 ? "Cliquez le second point de la ligne."
      : "Ligne tracée. Vérifiez le côté intérieur, puis validez.";
  }

  // Envoie une action ; l'image et le nombre de points viennent du serveur.
  function agir(corps, suite) {
    calib.enCours = true;
    majOutils();
    return fetch("/calibration/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corps)
    }).then(function (r) { return r.json(); }).then(function (rep) {
      calib.enCours = false;
      if (typeof rep.points === "number") { calib.points = rep.points; }
      if (rep.version !== undefined) { rafraichirImage(rep.version); }
      majOutils();
      if (!rep.accepte) { dire(rep.message, true); return; }
      dire("");
      if (suite) { suite(rep); }
    }).catch(function () {
      calib.enCours = false;
      dire("Serveur injoignable : l'action n'a pas été envoyée.", true);
      majOutils();
    });
  }

  function lancer(rep) { dire(rep.message, false); montrer("demo"); }

  toile.addEventListener("click", function (ev) {
    if ($("outils-ligne").hidden) { return; }
    var rect = toile.getBoundingClientRect();
    // Coordonnées normalisées dans l'image affichée : le serveur les rapporte
    // à la taille de l'image de la fenêtre de sélection.
    agir({
      action: "clic",
      x: Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (ev.clientY - rect.top) / rect.height))
    });
  });

  function traceActif() { return vueCourante === "calibration" && !$("outils-ligne").hidden; }

  $("btn-ligne").addEventListener("click", function () {
    agir({ action: "ligne" }, function () {
      $("choix-mode").hidden = true;
      $("outils-ligne").hidden = false;
    });
  });

  function inverser() {
    if (calib.points < 2) { return; }
    agir({ action: "inverser" });
  }

  function recommencer() { agir({ action: "recommencer" }); }

  function retour() {
    // Échap : retour à la question « ligne ou pas », ligne effacée.
    agir({ action: "retour" }, function () {
      $("outils-ligne").hidden = true;
      $("choix-mode").hidden = false;
      $("btn-ligne").focus();
    });
  }

  function valider() {
    if (calib.points < 2) {
      dire("Placez d'abord les deux extrémités de la ligne.", true);
      return;
    }
    if (calib.enCours) { return; }
    agir({ action: "valider" }, lancer);
  }

  var actions = { valider: valider, recommencer: recommencer, inverser: inverser, retour: retour };

  $("btn-inverser").addEventListener("click", inverser);
  $("btn-recommencer").addEventListener("click", recommencer);
  $("btn-retour").addEventListener("click", retour);
  $("btn-valider").addEventListener("click", valider);

  // Mêmes touches que la fenêtre OpenCV : C, G, I, Échap (voir interface.js).
  document.addEventListener("keydown", function (ev) {
    if (!traceActif()) { return; }
    var action = Interface.actionPourTouche(ev);
    if (!action) { return; }
    ev.preventDefault();
    actions[action]();
  });

  $("btn-sans-ligne").addEventListener("click", function () {
    agir({ action: "sans_ligne" }, lancer);
  });

  // ---------------------------------------------------------------------------
  // Démonstration
  // ---------------------------------------------------------------------------
  var demoLancee = false;
  var termine = false;
  var relecture = false;

  function afficherEtat(valeur) {
    $("etat").setAttribute("data-etat", valeur);
    $("etat-texte").textContent = Interface.texteEtat(valeur, relecture);
  }

  function afficher(nom, valeur) {
    // null = sans objet (mode sans ligne) : « Non assigné », texte atténué.
    var t = Interface.texteCompteur(valeur);
    $(nom).textContent = t.texte;
    $(nom).classList.toggle("non-assigne", t.attenue);
  }

  function demarrerDemo() {
    if (demoLancee) { return; }
    demoLancee = true;
    var cadre = $("cadre-flux"), flux = $("flux");
    // Espace réservé aux proportions de la vidéo jusqu'à la première image.
    cadre.classList.add("vide");
    if (calib.info && calib.info.largeur && calib.info.hauteur) {
      cadre.style.setProperty("--ratio", calib.info.largeur + " / " + calib.info.hauteur);
    }
    flux.src = "/video";
    var attente = setInterval(function () {
      if (flux.naturalWidth > 0) { cadre.classList.remove("vide"); clearInterval(attente); }
    }, 300);
  }

  var source = new EventSource("/stats");
  source.onmessage = function (m) {
    var v = JSON.parse(m.data);
    if (v.phase === "demonstration") { montrer("demo"); }
    afficher("presentes", v.presentes);
    afficher("entrees", v.entrees);
    afficher("sorties", v.sorties);
    afficher("nouvelles", v.nouvelles);
    $("mode-texte").textContent = Interface.texteMode(v.sans_ligne);
    // Avant la première image : « Chargement des modèles… », puis « Démarrage… ».
    if (v.etape) { $("demarrage").textContent = Interface.texteEtape(v.etape); }
    if (v.relecture && !relecture) { document.title = "Relecture : " + document.title; }
    relecture = !!v.relecture;
    termine = v.etat === "termine";
    afficherEtat(v.etat);
  };
  source.onerror = function () { if (!termine && vueCourante === "demo") { afficherEtat("hors_ligne"); } };

  // ---------------------------------------------------------------------------
  // Phase initiale
  // ---------------------------------------------------------------------------
  function interroger() {
    fetch("/calibration").then(function (r) { return r.json(); }).then(function (info) {
      calib.info = info;
      if (info.phase === "calibration") { montrer("calibration"); }
      else if (info.phase === "demonstration") { montrer("demo"); }
      else { montrer("attente"); setTimeout(interroger, 400); }
    }).catch(function () { setTimeout(interroger, 1000); });
  }
  interroger();
})();
