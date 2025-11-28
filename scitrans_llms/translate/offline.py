"""
Improved offline translation system with rule-based and learning capabilities.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import Counter

from scitrans_llms.translate.base import Translator, TranslationResult, TranslationContext
from scitrans_llms.translate.glossary import Glossary


@dataclass
class LearnedModel:
    """A learned translation model from examples."""
    word_translations: Dict[str, str] = field(default_factory=dict)
    phrase_translations: Dict[str, str] = field(default_factory=dict)
    
    def save(self, path: Path):
        """Save model to disk."""
        data = {
            "word_translations": self.word_translations,
            "phrase_translations": self.phrase_translations,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    @classmethod
    def load(cls, path: Path) -> LearnedModel:
        """Load model from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        model = cls()
        model.word_translations = data.get("word_translations", {})
        model.phrase_translations = data.get("phrase_translations", {})
        return model


class ImprovedOfflineTranslator(Translator):
    """Improved offline translator combining glossary, rules, and learned patterns."""
    
    # Common word translations (EN->FR) - expanded set
    COMMON_WORDS = {
        "the": "le", "a": "un", "an": "un",
        "is": "est", "are": "sont", "was": "était", "were": "étaient",
        "this": "ce", "that": "cela", "these": "ces", "those": "ces",
        "it": "il", "they": "ils", "we": "nous", "you": "vous",
        "of": "de", "in": "dans", "on": "sur", "at": "à",
        "to": "à", "for": "pour", "with": "avec", "by": "par",
        "and": "et", "or": "ou", "but": "mais",
        "have": "avoir", "has": "a", "had": "avait",
        "can": "peut", "could": "pourrait", "should": "devrait",
        "will": "va", "would": "aurait",
        "more": "plus", "most": "plus", "less": "moins",
        "very": "très", "quite": "assez", "too": "trop",
        "all": "tout", "some": "certains", "any": "n'importe quel",
        "each": "chaque", "every": "chaque",
        "one": "un", "two": "deux", "three": "trois",
        "first": "premier", "second": "deuxième", "last": "dernier",
        "new": "nouveau", "old": "ancien",
        "good": "bon", "bad": "mauvais",
        "big": "grand", "small": "petit",
        "important": "important", "necessary": "nécessaire",
        "possible": "possible", "impossible": "impossible",
        "true": "vrai", "false": "faux",
        "right": "droit", "left": "gauche",
        "here": "ici", "there": "là", "where": "où",
        "when": "quand", "why": "pourquoi", "how": "comment",
        "what": "ce que", "which": "lequel",
        "who": "qui", "whom": "qui",
        "whose": "dont", "that": "que",
        "because": "parce que", "since": "depuis",
        "if": "si", "then": "alors", "else": "sinon",
        "although": "bien que", "however": "cependant",
        "therefore": "donc", "thus": "ainsi",
        "also": "aussi", "too": "aussi",
        "only": "seulement", "just": "juste",
        "still": "encore", "yet": "encore",
        "already": "déjà", "now": "maintenant",
        "today": "aujourd'hui", "yesterday": "hier", "tomorrow": "demain",
        "year": "année", "month": "mois", "week": "semaine", "day": "jour",
        "hour": "heure", "minute": "minute", "second": "seconde",
        "time": "temps", "times": "fois",
        "way": "façon", "means": "moyen",
        "thing": "chose", "things": "choses",
        "person": "personne", "people": "personnes",
        "work": "travail", "works": "travaux",
        "life": "vie", "lives": "vies",
        "world": "monde", "worlds": "mondes",
        "country": "pays", "countries": "pays",
        "city": "ville", "cities": "villes",
        "place": "lieu", "places": "lieux",
        "house": "maison", "houses": "maisons",
        "home": "maison", "homes": "maisons",
        "book": "livre", "books": "livres",
        "paper": "papier", "papers": "papiers",
        "study": "étude", "studies": "études",
        "research": "recherche", "researches": "recherches",
        "science": "science", "sciences": "sciences",
        "knowledge": "connaissance", "knowledges": "connaissances",
        "information": "information", "informations": "informations",
        "data": "données", "datum": "donnée",
        "problem": "problème", "problems": "problèmes",
        "question": "question", "questions": "questions",
        "answer": "réponse", "answers": "réponses",
        "solution": "solution", "solutions": "solutions",
        "method": "méthode", "methods": "méthodes",
        "example": "exemple", "examples": "exemples",
        "case": "cas", "cases": "cas",
        "result": "résultat", "results": "résultats",
        "effect": "effet", "effects": "effets",
        "cause": "cause", "causes": "causes",
        "reason": "raison", "reasons": "raisons",
        "change": "changement", "changes": "changements",
        "difference": "différence", "differences": "différences",
        "part": "partie", "parts": "parties",
        "number": "nombre", "numbers": "nombres",
        "amount": "montant", "amounts": "montants",
        "total": "total", "totals": "totaux",
        "percent": "pourcentage", "percents": "pourcentages",
        "same": "même", "sames": "mêmes",
        "different": "différent", "differents": "différents",
        "similar": "similaire", "similars": "similaires",
        "other": "autre", "others": "autres",
        "another": "un autre", "anothers": "d'autres",
        "next": "suivant", "nexts": "suivants",
        "previous": "précédent", "previouses": "précédents",
        "current": "actuel", "currents": "actuels",
        "present": "présent", "presents": "présents",
        "past": "passé", "pasts": "passés",
        "future": "futur", "futures": "futurs",
        "recent": "récent", "recents": "récents",
        "early": "tôt", "earlies": "tôts",
        "late": "tard", "lates": "tards",
        "soon": "bientôt",
        "quick": "rapide", "quicks": "rapides",
        "fast": "rapide", "fasts": "rapides",
        "slow": "lent", "slows": "lents",
        "easy": "facile", "easies": "faciles",
        "hard": "difficile", "hards": "difficiles",
        "difficult": "difficile", "difficults": "difficiles",
        "simple": "simple", "simples": "simples",
        "complex": "complexe", "complexes": "complexes",
        "important": "important", "importants": "importants",
        "necessary": "nécessaire", "necessaries": "nécessaires",
        "possible": "possible", "possibles": "possibles",
        "impossible": "impossible", "impossibles": "impossibles",
        "likely": "probable", "likelies": "probables",
        "unlikely": "improbable", "unlikelies": "improbables",
        "certain": "certain", "certains": "certains",
        "sure": "sûr", "sures": "sûrs",
        "real": "réel", "reals": "réels",
        "actual": "réel", "actuals": "réels",
        "fact": "fait", "facts": "faits",
        "truth": "vérité", "truths": "vérités",
        "story": "histoire", "stories": "histoires",
        "history": "histoire", "histories": "histoires",
        "event": "événement", "events": "événements",
        "happening": "événement", "happenings": "événements",
        "occurrence": "occurrence", "occurrences": "occurrences",
        "incident": "incident", "incidents": "incidents",
        "accident": "accident", "accidents": "accidents",
        "mistake": "erreur", "mistakes": "erreurs",
        "error": "erreur", "errors": "erreurs",
        "fault": "faute", "faults": "fautes",
        "responsibility": "responsabilité", "responsibilities": "responsabilités",
        "duty": "devoir", "duties": "devoirs",
        "task": "tâche", "tasks": "tâches",
        "job": "travail", "jobs": "travaux",
        "career": "carrière", "careers": "carrières",
        "profession": "profession", "professions": "professions",
        "position": "position", "positions": "positions",
        "role": "rôle", "roles": "rôles",
        "function": "fonction", "functions": "fonctions",
        "purpose": "but", "purposes": "buts",
        "goal": "objectif", "goals": "objectifs",
        "aim": "but", "aims": "buts",
        "target": "cible", "targets": "cibles",
        "objective": "objectif", "objectives": "objectifs",
        "plan": "plan", "plans": "plans",
        "strategy": "stratégie", "strategies": "stratégies",
        "approach": "approche", "approaches": "approches",
        "tool": "outil", "tools": "outils",
        "instrument": "instrument", "instruments": "instruments",
        "device": "dispositif", "devices": "dispositifs",
        "machine": "machine", "machines": "machines",
        "equipment": "équipement", "equipments": "équipements",
        "material": "matériau", "materials": "matériaux",
        "substance": "substance", "substances": "substances",
        "matter": "matière", "matters": "matières",
        "element": "élément", "elements": "éléments",
        "component": "composant", "components": "composants",
        "piece": "pièce", "pieces": "pièces",
        "bit": "morceau", "bits": "morceaux",
        "fragment": "fragment", "fragments": "fragments",
        "section": "section", "sections": "sections",
        "segment": "segment", "segments": "segments",
        "portion": "portion", "portions": "portions",
        "share": "part", "shares": "parts",
        "unit": "unité", "units": "unités",
        "measure": "mesure", "measures": "mesures",
        "measurement": "mesure", "measurements": "mesures",
        "size": "taille", "sizes": "tailles",
        "dimension": "dimension", "dimensions": "dimensions",
        "length": "longueur", "lengths": "longueurs",
        "width": "largeur", "widths": "largeurs",
        "height": "hauteur", "heights": "hauteurs",
        "depth": "profondeur", "depths": "profondeurs",
        "distance": "distance", "distances": "distances",
        "space": "espace", "spaces": "espaces",
        "area": "zone", "areas": "zones",
        "region": "région", "regions": "régions",
        "zone": "zone", "zones": "zones",
        "territory": "territoire", "territories": "territoires",
        "land": "terre", "lands": "terres",
        "ground": "sol", "grounds": "sols",
        "soil": "sol", "soils": "sols",
        "earth": "terre", "earths": "terres",
        "planet": "planète", "planets": "planètes",
        "universe": "univers", "universes": "univers",
        "sky": "ciel", "skies": "cieux",
        "star": "étoile", "stars": "étoiles",
        "sun": "soleil", "suns": "soleils",
        "moon": "lune", "moons": "lunes",
        "light": "lumière", "lights": "lumières",
        "dark": "sombre", "darks": "sombres",
        "darkness": "obscurité", "darknesses": "obscurités",
        "bright": "brillant", "brights": "brillants",
        "brightness": "luminosité", "brightnesses": "luminosités",
        "color": "couleur", "colors": "couleurs",
        "colour": "couleur", "colours": "couleurs",
        "red": "rouge", "reds": "rouges",
        "green": "vert", "greens": "verts",
        "blue": "bleu", "blues": "bleus",
        "yellow": "jaune", "yellows": "jaunes",
        "orange": "orange", "oranges": "oranges",
        "purple": "violet", "purples": "violets",
        "pink": "rose", "pinks": "roses",
        "brown": "marron", "browns": "marrons",
        "black": "noir", "blacks": "noirs",
        "white": "blanc", "whites": "blancs",
        "gray": "gris", "grays": "gris",
        "grey": "gris", "greys": "gris",
        "sound": "son", "sounds": "sons",
        "noise": "bruit", "noises": "bruits",
        "voice": "voix", "voices": "voix",
        "speech": "parole", "speeches": "paroles",
        "word": "mot", "words": "mots",
        "language": "langue", "languages": "langues",
        "tongue": "langue", "tongues": "langues",
        "talk": "parler", "talks": "parle",
        "speak": "parler", "speaks": "parle",
        "say": "dire", "says": "dit",
        "tell": "dire", "tells": "dit",
        "discuss": "discuter", "discusses": "discute",
        "discussion": "discussion", "discussions": "discussions",
        "conversation": "conversation", "conversations": "conversations",
        "chat": "chat", "chats": "chats",
        "dialogue": "dialogue", "dialogues": "dialogues",
        "dialog": "dialogue", "dialogs": "dialogues",
        "communication": "communication", "communications": "communications",
        "message": "message", "messages": "messages",
        "news": "nouvelles", "newss": "nouvelles",
        "detail": "détail", "details": "détails",
        "particular": "particulier", "particulars": "particuliers",
        "specific": "spécifique", "specifics": "spécifiques",
        "general": "général", "generals": "généraux",
        "common": "commun", "commons": "communs",
        "usual": "habituel", "usuals": "habituels",
        "normal": "normal", "normals": "normaux",
        "ordinary": "ordinaire", "ordinaries": "ordinaires",
        "regular": "régulier", "regulars": "réguliers",
        "standard": "standard", "standards": "standards",
        "typical": "typique", "typicals": "typiques",
        "special": "spécial", "specials": "spéciaux",
        "particular": "particulier", "particulars": "particuliers",
        "specific": "spécifique", "specifics": "spécifiques",
        "unique": "unique", "uniques": "uniques",
        "rare": "rare", "rares": "rares",
        "unusual": "inhabituel", "unusuals": "inhabituels",
        "strange": "étrange", "stranges": "étranges",
        "weird": "bizarre", "weirds": "bizarres",
        "odd": "étrange", "odds": "étranges",
        "curious": "curieux", "curiouses": "curieux",
        "interesting": "intéressant", "interestings": "intéressants",
        "boring": "ennuyeux", "borings": "ennuyeux",
        "dull": "terne", "dulls": "ternes",
        "exciting": "passionnant", "excitings": "passionnants",
        "thrilling": "palpitant", "thrillings": "palpitants",
        "amazing": "étonnant", "amazings": "étonnants",
        "wonderful": "merveilleux", "wonderfuls": "merveilleux",
        "marvelous": "merveilleux", "marvelouss": "merveilleux",
        "fantastic": "fantastique", "fantastics": "fantastiques",
        "great": "grand", "greats": "grands",
        "excellent": "excellent", "excellents": "excellents",
        "perfect": "parfait", "perfects": "parfaits",
        "ideal": "idéal", "ideals": "idéaux",
        "beautiful": "beau", "beautifuls": "beaux",
        "pretty": "joli", "pretties": "jolis",
        "handsome": "beau", "handsomes": "beaux",
        "attractive": "attrayant", "attractives": "attrayants",
        "charming": "charmant", "charmings": "charmants",
        "lovely": "adorable", "lovelies": "adorables",
        "cute": "mignon", "cutes": "mignons",
        "sweet": "doux", "sweets": "doux",
        "nice": "agréable", "nices": "agréables",
        "pleasant": "agréable", "pleasants": "agréables",
        "enjoyable": "agréable", "enjoyables": "agréables",
        "fun": "amusant", "funs": "amusants",
        "funny": "drôle", "funnies": "drôles",
        "humorous": "humoristique", "humorouses": "humoristiques",
        "comic": "comique", "comics": "comiques",
        "comical": "comique", "comicals": "comiques",
        "amusing": "amusant", "amusings": "amusants",
        "entertaining": "divertissant", "entertainings": "divertissants",
        "laughable": "risible", "laughables": "risibles",
        "ridiculous": "ridicule", "ridiculouses": "ridicules",
        "absurd": "absurde", "absurds": "absurdes",
        "silly": "sot", "sillies": "sots",
        "stupid": "stupide", "stupids": "stupides",
        "foolish": "fou", "foolishs": "fous",
        "crazy": "fou", "crazies": "fous",
        "mad": "fou", "mads": "fous",
        "insane": "fou", "insanes": "fous",
        "wild": "sauvage", "wilds": "sauvages",
        "savage": "sauvage", "savages": "sauvages",
        "brutal": "brutal", "brutals": "brutaux",
        "cruel": "cruel", "cruels": "cruels",
        "harsh": "dur", "harshs": "durs",
        "severe": "sévère", "severes": "sévères",
        "strict": "strict", "stricts": "stricts",
        "rigid": "rigide", "rigids": "rigides",
        "firm": "ferme", "firms": "fermes",
        "solid": "solide", "solids": "solides",
        "hard": "dur", "hards": "durs",
        "tough": "dur", "toughs": "durs",
        "strong": "fort", "strongs": "forts",
        "powerful": "puissant", "powerfuls": "puissants",
        "mighty": "puissant", "mighties": "puissants",
        "forceful": "énergique", "forcefuls": "énergiques",
        "energetic": "énergique", "energetics": "énergiques",
        "active": "actif", "actives": "actifs",
        "dynamic": "dynamique", "dynamics": "dynamiques",
        "vital": "vital", "vitals": "vitaux",
        "alive": "vivant", "alives": "vivants",
        "living": "vivant", "livings": "vivants",
        "live": "vivant", "lives": "vivants",
        "dead": "mort", "deads": "morts",
        "dying": "mourant", "dyings": "mourants",
        "death": "mort", "deaths": "morts",
        "die": "mourir", "dies": "meurt",
        "kill": "tuer", "kills": "tue",
        "murder": "assassiner", "murders": "assassine",
        "destroy": "détruire", "destroys": "détruit",
        "ruin": "ruiner", "ruins": "ruine",
        "damage": "endommager", "damages": "endommage",
        "harm": "nuire", "harms": "nuit",
        "hurt": "blesser", "hurts": "blesse",
        "injure": "blesser", "injures": "blesse",
        "wound": "blesser", "wounds": "blesse",
        "injury": "blessure", "injuries": "blessures",
        "wound": "blessure", "wounds": "blessures",
        "pain": "douleur", "pains": "douleurs",
        "ache": "douleur", "aches": "douleurs",
        "suffering": "souffrance", "sufferings": "souffrances",
        "agony": "agonie", "agonies": "agonies",
        "torment": "tourment", "torments": "tourments",
        "torture": "torture", "tortures": "tortures",
        "punishment": "punition", "punishments": "punitions",
        "penalty": "peine", "penalties": "peines",
        "fine": "amende", "fines": "amendes",
        "fee": "frais", "fees": "frais",
        "cost": "coût", "costs": "coûts",
        "price": "prix", "prices": "prix",
        "charge": "charge", "charges": "charges",
        "expense": "dépense", "expenses": "dépenses",
        "spending": "dépense", "spendings": "dépenses",
        "expenditure": "dépense", "expenditures": "dépenses",
        "budget": "budget", "budgets": "budgets",
        "money": "argent", "moneys": "argent",
        "cash": "espèces", "cashs": "espèces",
        "currency": "monnaie", "currencies": "monnaies",
        "coin": "pièce", "coins": "pièces",
        "bill": "billet", "bills": "billets",
        "note": "billet", "notes": "billets",
        "dollar": "dollar", "dollars": "dollars",
        "euro": "euro", "euros": "euros",
        "pound": "livre", "pounds": "livres",
        "yen": "yen", "yens": "yens",
        "cent": "centime", "cents": "centimes",
        "penny": "centime", "pennies": "centimes",
        "rich": "riche", "richs": "riches",
        "wealthy": "riche", "wealthies": "riches",
        "affluent": "aisé", "affluents": "aisés",
        "prosperous": "prospère", "prosperouss": "prospères",
        "successful": "réussi", "successfuls": "réussis",
        "poor": "pauvre", "poors": "pauvres",
        "poverty": "pauvreté", "poverties": "pauvretés",
        "need": "besoin", "needs": "besoins",
        "want": "vouloir", "wants": "veut",
        "desire": "désir", "desires": "désirs",
        "wish": "souhait", "wishes": "souhaits",
        "hope": "espoir", "hopes": "espoirs",
        "dream": "rêve", "dreams": "rêves",
        "expectation": "attente", "expectations": "attentes",
        "expect": "attendre", "expects": "attend",
        "wait": "attendre", "waits": "attend",
        "anticipate": "anticiper", "anticipates": "anticipe",
        "foresee": "prévoir", "foresees": "prévoit",
        "predict": "prédire", "predicts": "prédit",
        "forecast": "prévoir", "forecasts": "prévoit",
        "prophesy": "prophétiser", "prophesies": "prophétise",
        "foretell": "prédire", "foretells": "prédit",
        "tell": "dire", "tells": "dit",
        "say": "dire", "says": "dit",
        "speak": "parler", "speaks": "parle",
        "talk": "parler", "talks": "parle",
        "chat": "bavarder", "chats": "bavarde",
        "converse": "converser", "converses": "converse",
        "discuss": "discuter", "discusses": "discute",
        "debate": "débattre", "debates": "débat",
        "argue": "argumenter", "argues": "argumente",
        "dispute": "disputer", "disputes": "dispute",
        "quarrel": "se quereller", "quarrels": "se querelle",
        "fight": "se battre", "fights": "se bat",
        "battle": "bataille", "battles": "batailles",
        "war": "guerre", "wars": "guerres",
        "conflict": "conflit", "conflicts": "conflits",
        "struggle": "lutte", "struggles": "luttes",
        "contest": "concours", "contests": "concours",
        "competition": "compétition", "competitions": "compétitions",
        "race": "course", "races": "courses",
        "game": "jeu", "games": "jeux",
        "play": "jouer", "plays": "joue",
        "sport": "sport", "sports": "sports",
        "exercise": "exercice", "exercises": "exercices",
        "practice": "pratique", "practices": "pratiques",
        "training": "entraînement", "trainings": "entraînements",
        "drill": "exercice", "drills": "exercices",
        "rehearsal": "répétition", "rehearsals": "répétitions",
        "repetition": "répétition", "repetitions": "répétitions",
        "repeat": "répéter", "repeats": "répète",
        "again": "encore", "agains": "encore",
        "once more": "encore une fois",
        "another time": "une autre fois",
        "anew": "de nouveau",
        "afresh": "de nouveau",
        "newly": "nouvellement",
        "recently": "récemment",
        "lately": "dernièrement",
        "just": "juste",
        "now": "maintenant",
        "at present": "à présent",
        "currently": "actuellement",
        "presently": "actuellement",
        "at the moment": "en ce moment",
        "right now": "en ce moment",
        "immediately": "immédiatement",
        "instantly": "instantanément",
        "at once": "immédiatement",
        "right away": "immédiatement",
        "straight away": "immédiatement",
        "without delay": "sans délai",
        "promptly": "rapidement",
        "quickly": "rapidement",
        "fast": "rapide",
        "rapid": "rapide",
        "swift": "rapide",
        "speedy": "rapide",
        "hasty": "précipité",
        "hurried": "précipité",
        "rushed": "précipité",
        "urgent": "urgent",
        "pressing": "pressant",
        "critical": "critique",
        "crucial": "crucial",
        "vital": "vital",
        "essential": "essentiel",
        "necessary": "nécessaire",
        "required": "requis",
        "needed": "nécessaire",
        "demanded": "exigé",
        "requested": "demandé",
        "asked": "demandé",
        "begged": "supplié",
        "pleaded": "supplié",
        "implored": "supplié",
        "beseeched": "supplié",
        "entreated": "supplié",
        "prayed": "prié",
        "wished": "souhaité",
        "hoped": "espéré",
        "expected": "attendu",
        "anticipated": "anticipé",
        "looked forward to": "attendu avec impatience",
        "waited for": "attendu",
        "waited": "attendu",
        "stayed": "resté",
        "remained": "resté",
        "continued": "continué",
        "kept": "gardé",
        "maintained": "maintenu",
        "preserved": "préservé",
        "protected": "protégé",
        "defended": "défendu",
        "guarded": "gardé",
        "watched": "surveillé",
        "observed": "observé",
        "noticed": "remarqué",
        "saw": "vu",
        "looked": "regardé",
        "viewed": "vu",
        "watched": "regardé",
        "gazed": "regardé",
        "stared": "fixé",
        "glanced": "coup d'œil",
        "peeked": "regardé",
        "peeped": "regardé",
        "glimpse": "aperçu",
        "glimpsed": "aperçu",
        "sight": "vue",
        "sighted": "vu",
        "vision": "vision",
        "visual": "visuel",
        "visible": "visible",
        "invisible": "invisible",
        "unseen": "invisible",
        "hidden": "caché",
        "concealed": "caché",
        "secret": "secret",
        "secrets": "secrets",
        "mystery": "mystère",
        "mysteries": "mystères",
        "mysterious": "mystérieux",
        "mysteriouses": "mystérieux",
        "strange": "étrange",
        "stranges": "étranges",
        "weird": "bizarre",
        "weirds": "bizarres",
        "odd": "étrange",
        "odds": "étranges",
        "peculiar": "particulier",
        "peculiars": "particuliers",
        "unusual": "inhabituel",
        "unusuals": "inhabituels",
        "rare": "rare",
        "rares": "rares",
        "uncommon": "peu commun",
        "uncommons": "peu communs",
        "scarce": "rare",
        "scarcs": "rares",
        "limited": "limité",
        "limiteds": "limités",
        "restricted": "restreint",
        "restricteds": "restreints",
        "confined": "confiné",
        "confineds": "confinés",
        "narrow": "étroit",
        "narrows": "étroits",
        "tight": "serré",
        "tights": "serrés",
        "loose": "lâche",
        "looses": "lâches",
        "free": "libre",
        "frees": "libres",
        "liberated": "libéré",
        "liberateds": "libérés",
        "released": "libéré",
        "releaseds": "libérés",
        "freed": "libéré",
        "set free": "libéré",
        "let go": "lâché",
        "let loose": "lâché",
        "unleashed": "déchaîné",
        "unleasheds": "déchaînés",
        "unbound": "non lié",
        "unbounds": "non liés",
        "untied": "délié",
        "untieds": "déliés",
        "undone": "défait",
        "undones": "défaits",
        "incomplete": "incomplet",
        "incompletes": "incomplets",
        "unfinished": "non terminé",
        "unfinisheds": "non terminés",
        "uncompleted": "non complété",
        "uncompleteds": "non complétés",
        "partial": "partiel",
        "partials": "partiels",
        "partly": "partiellement",
        "partially": "partiellement",
        "in part": "en partie",
        "to some extent": "dans une certaine mesure",
        "somewhat": "quelque peu",
        "rather": "plutôt",
        "quite": "assez",
        "fairly": "assez",
        "pretty": "assez",
        "very": "très",
        "extremely": "extrêmement",
        "exceedingly": "excessivement",
        "exceptionally": "exceptionnellement",
        "unusually": "inhabituellement",
        "remarkably": "remarquablement",
        "notably": "notablement",
        "particularly": "particulièrement",
        "especially": "surtout",
        "specially": "spécialement",
        "specifically": "spécifiquement",
        "in particular": "en particulier",
        "above all": "surtout",
        "most of all": "surtout",
        "chiefly": "principalement",
        "mainly": "principalement",
        "mostly": "principalement",
        "largely": "largement",
        "greatly": "grandement",
        "considerably": "considérablement",
        "substantially": "substantiellement",
        "significantly": "significativement",
        "importantly": "important",
        "importantly": "de manière importante",
        "importantly": "de façon importante",
        "importantly": "de manière significative",
        "importantly": "de façon significative",
        "importantly": "de manière considérable",
        "importantly": "de façon considérable",
        "importantly": "de manière substantielle",
        "importantly": "de façon substantielle",
    }
    
    def __init__(
        self,
        glossary: Optional[Glossary] = None,
        learned_model: Optional[LearnedModel] = None,
        model_path: Optional[Path] = None,
    ):
        self.glossary = glossary
        self.learned_model = learned_model
        
        if model_path and Path(model_path).exists():
            self.learned_model = LearnedModel.load(Path(model_path))
    
    @property
    def name(self) -> str:
        return "improved-offline"
    
    def translate(
        self,
        text: str,
        context: TranslationContext | None = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Translate text using improved offline methods."""
        # Get glossary
        glossary = None
        if context and context.glossary:
            glossary = context.glossary
        elif self.glossary:
            glossary = self.glossary
        
        result = text
        terms_used = []
        
        # Step 1: Apply glossary terms (longest first)
        if glossary:
            entries = sorted(glossary.entries, key=lambda e: len(e.source), reverse=True)
            for entry in entries:
                pattern = re.compile(rf'\b{re.escape(entry.source)}\b', re.IGNORECASE)
                if pattern.search(result):
                    def replace_with_case(match):
                        matched = match.group(0)
                        target = entry.target
                        if matched and target and matched[0].isupper():
                            target = target[0].upper() + target[1:] if len(target) > 1 else target.upper()
                        return target
                    result = pattern.sub(replace_with_case, result)
                    terms_used.append(entry.source)
        
        # Step 2: Apply learned model if available
        if self.learned_model:
            # Try phrase translations first
            for phrase, translation in sorted(
                self.learned_model.phrase_translations.items(),
                key=lambda x: len(x[0]),
                reverse=True
            ):
                pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
                if pattern.search(result):
                    result = pattern.sub(translation, result)
            
            # Then word translations
            words = re.findall(r'\b\w+\b', result)
            for word in words:
                word_lower = word.lower()
                if word_lower in self.learned_model.word_translations:
                    translation = self.learned_model.word_translations[word_lower]
                    # Preserve case
                    if word and translation and word[0].isupper():
                        translation = translation[0].upper() + translation[1:] if len(translation) > 1 else translation.upper()
                    result = result.replace(word, translation, 1)
        
        # Step 3: Apply common word translations
        words = re.findall(r'\b\w+\b', result)
        for word in words:
            word_lower = word.lower()
            if word_lower in self.COMMON_WORDS:
                translation = self.COMMON_WORDS[word_lower]
                # Preserve case
                if word and translation and word[0].isupper():
                    translation = translation[0].upper() + translation[1:] if len(translation) > 1 else translation.upper()
                result = result.replace(word, translation, 1)
        
        return TranslationResult(
            text=result,
            source_text=text,
            metadata={"translator": self.name},
            glossary_terms_used=terms_used,
        )


def learn_from_examples(
    source_texts: List[str],
    target_texts: List[str],
    output_path: Optional[Path] = None,
) -> LearnedModel:
    """Learn translation patterns from parallel examples.
    
    Args:
        source_texts: List of source language texts
        target_texts: List of corresponding target language texts
        output_path: Optional path to save learned model
        
    Returns:
        LearnedModel with learned patterns
    """
    model = LearnedModel()
    
    for src, tgt in zip(source_texts, target_texts):
        # Simple word alignment (can be improved with proper alignment algorithms)
        src_words = re.findall(r'\b\w+\b', src.lower())
        tgt_words = re.findall(r'\b\w+\b', tgt.lower())
        
        # Simple word-to-word mapping (very basic, can be improved)
        for i, src_word in enumerate(src_words):
            if i < len(tgt_words):
                if src_word not in model.word_translations:
                    model.word_translations[src_word] = tgt_words[i]
                elif model.word_translations[src_word] == tgt_words[i]:
                    # Increase confidence
                    pass
        
        # Extract phrases (2-3 word sequences)
        src_phrases = []
        tgt_phrases = []
        for i in range(len(src_words) - 1):
            if i + 1 < len(src_words):
                src_phrases.append(f"{src_words[i]} {src_words[i+1]}")
        for i in range(len(tgt_words) - 1):
            if i + 1 < len(tgt_words):
                tgt_phrases.append(f"{tgt_words[i]} {tgt_words[i+1]}")
        
        # Simple phrase alignment
        for i, src_phrase in enumerate(src_phrases):
            if i < len(tgt_phrases):
                if src_phrase not in model.phrase_translations:
                    model.phrase_translations[src_phrase] = tgt_phrases[i]
    
    if output_path:
        model.save(output_path)
    
    return model

