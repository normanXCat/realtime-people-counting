// Interface web : calibration dans le navigateur, puis vue de démonstration.
// Aucune logique de comptage ici : la page envoie la ligne (ou « sans ligne »)
// au serveur, qui la valide comme la calibration OpenCV, puis affiche l'état publié.
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
  // Géométrie : même convention que geometry.signed_perpendicular_distance.
  // d(P) = ((px - x1) * dy - (py - y1) * dx) / |AB| ; P est à l'intérieur si
  // d(P) * signe(côté) > 0, avec signe(« positive ») = +1, signe(« negative ») = -1.
  // Le signe ne dépend pas de l'échelle d'affichage (coordonnées positives).
  // ---------------------------------------------------------------------------
  function signeCote(cote) { return cote === "positive" ? 1 : -1; }

  function directionInterieure(a, b, cote) {
    var dx = b.x - a.x, dy = b.y - a.y, n = Math.hypot(dx, dy) || 1;
    var s = signeCote(cote);
    return { x: s * dy / n, y: -s * dx / n };
  }

  // ---------------------------------------------------------------------------
  // Calibration
  // ---------------------------------------------------------------------------
  var calib = { info: null, points: [], cote: "negative", enCours: false, prete: false };
  var image = $("image-calibration");
  var toile = $("trace");
  var message = $("message");

  function dire(texte, erreur) {
    message.textContent = texte || "";
    message.classList.toggle("erreur", !!erreur);
  }

  function demarrerCalibration() {
    if (calib.prete) { return; }
    calib.prete = true;
    image.src = "/calibration/image?t=" + Date.now();
    image.onload = dessiner;
    if (window.ResizeObserver) { new ResizeObserver(dessiner).observe(image); }
    else { window.addEventListener("resize", dessiner); }
  }

  function taillesToile() {
    var r = window.devicePixelRatio || 1;
    var w = image.clientWidth, h = image.clientHeight;
    if (toile.width !== Math.round(w * r) || toile.height !== Math.round(h * r)) {
      toile.width = Math.round(w * r);
      toile.height = Math.round(h * r);
    }
    return { w: w, h: h, r: r };
  }

  function versPixels(p, t) { return { x: p.x * t.w, y: p.y * t.h }; }

  function dessiner() {
    var t = taillesToile();
    var ctx = toile.getContext("2d");
    ctx.setTransform(t.r, 0, 0, t.r, 0, 0);
    ctx.clearRect(0, 0, t.w, t.h);
    var accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#ffc800";
    var pts = calib.points.map(function (p) { return versPixels(p, t); });

    if (pts.length === 2) {
      var a = pts[0], b = pts[1];
      var dx = b.x - a.x, dy = b.y - a.y, n = Math.hypot(dx, dy);
      if (n > 0) {
        var ux = dx / n, uy = dy / n, grand = 4 * Math.hypot(t.w, t.h);
        var ext1 = { x: a.x - ux * grand, y: a.y - uy * grand };
        var ext2 = { x: b.x + ux * grand, y: b.y + uy * grand };
        var d = directionInterieure(a, b, calib.cote);

        // Côté intérieur légèrement teinté : demi-plan limité par la droite prolongée.
        ctx.beginPath();
        ctx.moveTo(ext1.x, ext1.y);
        ctx.lineTo(ext2.x, ext2.y);
        ctx.lineTo(ext2.x + d.x * grand, ext2.y + d.y * grand);
        ctx.lineTo(ext1.x + d.x * grand, ext1.y + d.y * grand);
        ctx.closePath();
        ctx.fillStyle = "rgba(255, 200, 0, 0.16)";
        ctx.fill();

        // Prolongement en pointillés sur toute l'image.
        ctx.save();
        ctx.setLineDash([10, 8]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.beginPath();
        ctx.moveTo(ext1.x, ext1.y);
        ctx.lineTo(ext2.x, ext2.y);
        ctx.stroke();
        ctx.restore();

        // Segment tracé.
        ctx.lineWidth = 4;
        ctx.strokeStyle = accent;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();

        // Étiquettes des deux côtés.
        var m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, off = 0.1 * Math.min(t.w, t.h);
        etiquette(ctx, "Intérieur", m.x + d.x * off, m.y + d.y * off, accent);
        etiquette(ctx, "Extérieur", m.x - d.x * off, m.y - d.y * off, "#ffffff");
      }
    }

    pts.forEach(function (p, i) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = accent;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#111";
      ctx.stroke();
      etiquette(ctx, String(i + 1), p.x + 16, p.y - 16, "#ffffff");
    });
  }

  function etiquette(ctx, texte, x, y, couleur) {
    ctx.font = "600 16px system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    var w = ctx.measureText(texte).width + 16;
    ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
    ctx.fillRect(x - w / 2, y - 14, w, 28);
    ctx.fillStyle = couleur;
    ctx.fillText(texte, x, y);
  }

  function majOutils() {
    var n = calib.points.length;
    $("btn-inverser").disabled = n < 2;
    $("btn-valider").disabled = n < 2 || calib.enCours;
    $("consigne").textContent = n === 0 ? "Cliquez le premier point de la ligne."
      : n === 1 ? "Cliquez le second point de la ligne."
      : "Ligne tracée. Vérifiez le côté intérieur, puis validez.";
    dessiner();
  }

  toile.addEventListener("click", function (ev) {
    if ($("outils-ligne").hidden) { return; }
    if (calib.points.length >= 2) {
      dire("Deux points sont déjà placés : G (Recommencer) pour tracer une autre ligne.", true);
      return;
    }
    var rect = toile.getBoundingClientRect();
    // Coordonnées normalisées : indépendantes de la taille d'affichage,
    // rapportées par le serveur à la taille réelle de l'image.
    var p = {
      x: Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (ev.clientY - rect.top) / rect.height))
    };
    calib.points.push(p);
    dire("");
    majOutils();
  });

  function traceActif() { return vueCourante === "calibration" && !$("outils-ligne").hidden; }

  $("btn-ligne").addEventListener("click", function () {
    $("choix-mode").hidden = true;
    $("outils-ligne").hidden = false;
    calib.points = [];
    calib.cote = (calib.info && calib.info.cote_interieur) || "negative";
    dire("");
    majOutils();
  });

  function inverser() {
    if (calib.points.length < 2) { return; }
    calib.cote = calib.cote === "positive" ? "negative" : "positive";
    dessiner();
  }

  function recommencer() {
    calib.points = [];
    dire("");
    majOutils();
  }

  function retour() {
    // Échap : retour à la question « ligne ou pas », ligne effacée.
    calib.points = [];
    dire("");
    $("outils-ligne").hidden = true;
    $("choix-mode").hidden = false;
    dessiner();
    $("btn-ligne").focus();
  }

  function valider() {
    if (calib.points.length < 2) {
      dire("Placez d'abord les deux extrémités de la ligne.", true);
      return;
    }
    if (calib.enCours) { return; }
    var a = calib.points[0], b = calib.points[1];
    envoyer({
      mode: "ligne",
      p1: [a.x, a.y],
      p2: [b.x, b.y],
      cote_interieur: calib.cote,
      largeur_affichee: image.clientWidth
    });
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

  function envoyer(corps) {
    calib.enCours = true;
    majOutils();
    return fetch("/calibration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corps)
    }).then(function (r) { return r.json(); }).then(function (rep) {
      calib.enCours = false;
      if (rep.accepte) { dire(rep.message, false); montrer("demo"); }
      else { dire(rep.message, true); majOutils(); }
    }).catch(function () {
      calib.enCours = false;
      dire("Serveur injoignable : la calibration n'a pas été envoyée.", true);
      majOutils();
    });
  }

  $("btn-sans-ligne").addEventListener("click", function () {
    envoyer({ mode: "sans_ligne" });
  });

  // ---------------------------------------------------------------------------
  // Démonstration
  // ---------------------------------------------------------------------------
  var demoLancee = false;
  var termine = false;
  var textesEtat = { en_cours: "En cours", termine: "Terminé", hors_ligne: "Connexion perdue" };

  function afficherEtat(valeur) {
    $("etat").setAttribute("data-etat", valeur);
    $("etat-texte").textContent = textesEtat[valeur] || valeur;
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
