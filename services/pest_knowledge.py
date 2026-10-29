"""Crop and disease metadata for the 38 PlantVillage classes supported by the
MobileNetV2 pest classifier. Guidance is general agronomic practice; chemical
products are named only by active-ingredient family and must always be applied
according to the label registered for the specific crop and region."""

CROP_DISPLAY_NAMES = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn (maize)",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Bell pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}

CONDITION_DISPLAY_NAMES = {
    "Apple_scab": "Apple scab",
    "Black_rot": "Black rot",
    "Cedar_apple_rust": "Cedar apple rust",
    "healthy": "Healthy leaf",
    "Powdery_mildew": "Powdery mildew",
    "Cercospora_leaf_spot Gray_leaf_spot": "Cercospora / gray leaf spot",
    "Common_rust_": "Common rust",
    "Northern_Leaf_Blight": "Northern leaf blight",
    "Esca_(Black_Measles)": "Esca (black measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis leaf spot)",
    "Haunglongbing_(Citrus_greening)": "Huanglongbing (citrus greening)",
    "Bacterial_spot": "Bacterial spot",
    "Early_blight": "Early blight",
    "Late_blight": "Late blight",
    "Leaf_scorch": "Leaf scorch",
    "Leaf_Mold": "Leaf mould",
    "Septoria_leaf_spot": "Septoria leaf spot",
    "Spider_mites Two-spotted_spider_mite": "Two-spotted spider mite damage",
    "Target_Spot": "Target spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    "Tomato_mosaic_virus": "Tomato mosaic virus",
}

CHEMICAL_APPLICATION_NOTE = (
    "Apply only products registered for this crop in your region, at the label rate, "
    "and confirm the diagnosis with your local extension service before spraying."
)

TREATMENTS = {
    "Apple___Apple_scab": {
        "root_cause": (
            "Fungal infection by Venturia inaequalis. Spores overwinter in fallen leaves, "
            "canes and buds and infect new growth during wet spring weather."
        ),
        "treatment": (
            "Rake and remove fallen leaves, prune out crowded growth to speed drying, and run a "
            "protectant spray programme with captan or mancozeb from bud break through the wet "
            "period. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Plant scab-resistant cultivars, remove leaf litter from the orchard floor, keep "
            "canopies open and monitor weather-driven scab warnings each spring."
        ),
    },
    "Apple___Black_rot": {
        "root_cause": (
            "Fungal infection by Botryosphaeria obtusa. Cankers on wood and mummified fruit are "
            "the source of spores, which spread during warm, wet weather and also enter wounds."
        ),
        "treatment": (
            "Prune and destroy cankered wood and mummified fruit, remove dead branches, and apply "
            "a registered fungicide (captan or thiophanate-methyl families) during wet periods. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Keep orchard sanitation strict, avoid tree wounds, control insect and fire blight "
            "injury and maintain balanced nutrition to reduce stress."
        ),
    },
    "Apple___Cedar_apple_rust": {
        "root_cause": (
            "Fungal infection by Gymnosporangium juniperi-virginianae. It requires juniper or "
            "cedar as an alternate host; rain releases spores from juniper galls onto apple foliage."
        ),
        "treatment": (
            "Remove nearby juniper galls where practical and protect foliage with myclobutanil from "
            "pink bud until about six weeks after bloom. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Choose rust-resistant apple cultivars and keep juniper/cedar hosts several hundred "
            "metres away from the orchard where possible."
        ),
    },
    "Apple___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": (
            "No curative action needed for this leaf. Keep the routine calendar of dormant pruning, "
            "balanced fertiliser and spring fungicide protection if scab or rust is common locally."
        ),
        "prevention": (
            "Continue weekly scouting of new growth, remove leaf litter, keep canopies pruned for "
            "airflow and maintain soil and nutrient management through the season."
        ),
    },
    "Blueberry___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Keep soil pH in the 4.5-5.5 range, maintain mulch and drip irrigation, prune out old "
            "canes and scout leaves for mummy berry and leaf spot through the season."
        ),
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "root_cause": (
            "Fungal infection by Podosphaera clandestina, favoured by warm days, cool nights and "
            "high humidity in dense, shaded canopies."
        ),
        "treatment": (
            "Improve airflow by pruning, reduce excessive nitrogen, remove heavily affected shoots "
            "and use a registered sulphur or potassium bicarbonate product when symptoms first "
            "appear. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Train open canopies, avoid overcrowding and excessive nitrogen, and monitor new growth "
            "in humid weather."
        ),
    },
    "Cherry_(including_sour)___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Maintain annual pruning for open canopies, control weeds around the tree base and "
            "scout for aphids and leaf spot during humid periods."
        ),
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "root_cause": (
            "Fungal infection by Cercospora zeae-maydis. The fungus survives on maize residue and "
            "infections build up during prolonged leaf wetness and high humidity after tasselling."
        ),
        "treatment": (
            "There is no rescue treatment for existing lesions. If disease pressure is high on a "
            "susceptible hybrid, a strobilurin or triazole fungicide applied between tasselling and "
            "silking can protect upper leaves. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Rotate away from maize, manage residue with tillage where appropriate, choose hybrids "
            "with gray leaf spot resistance and avoid late planting in continuous maize fields."
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "root_cause": (
            "Fungal infection by Puccinia sorghi. Rust spores blow in from southern regions each "
            "season and germinate on leaves during cool, humid weather with heavy dew."
        ),
        "treatment": (
            "Rust usually appears after flowering and rarely justifies spraying; a foliar fungicide "
            "is only warranted on very susceptible hybrids when pustules cover upper leaves before "
            "grain fill. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Use hybrids with rust resistance, plant on time and monitor fields from mid-season "
            "when humid nights favour rust development."
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "root_cause": (
            "Fungal infection by Exserohilum turcicum. The pathogen survives on crop residue and "
            "spreads during moderate temperatures with extended leaf wetness."
        ),
        "treatment": (
            "Destroy/incorporate residue after harvest and, on susceptible hybrids with lesions on "
            "upper leaves before tasselling, apply a registered triazole or strobilurin fungicide. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Rotate crops, manage residue, plant resistant hybrids and avoid continuous maize "
            "cropping in the same field."
        ),
    },
    "Corn_(maize)___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Rotate maize with other crops, keep balanced nitrogen and potassium fertility, and "
            "scout for leaf blights from tasselling onward in humid weather."
        ),
    },
    "Grape___Black_rot": {
        "root_cause": (
            "Fungal infection by Guignardia bidwellii. Spores are released from mummified berries "
            "and infected canes during wet weather from bud break to fruit set."
        ),
        "treatment": (
            "Remove and destroy mummified berries and infected canes during pruning, keep the "
            "canopy open and follow a protectant programme (mancozeb, captan or myclobutanil "
            "families) from bud break through fruit set. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Practise strict vineyard sanitation, train vines for airflow, remove volunteer vines "
            "and wild grapes nearby and scout from bud break."
        ),
    },
    "Grape___Esca_(Black_Measles)": {
        "root_cause": (
            "A wood disease complex of several fungi that colonise pruning wounds and trunk tissue, "
            "causing sudden leaf scorch, tiger-striped leaves and fruit spotting in older vines."
        ),
        "treatment": (
            "There is no cure. Mark and remove severely affected vines, prune out infected wood into "
            "healthy tissue, avoid large pruning wounds and protect fresh cuts with a pruning "
            "wound sealant."
        ),
        "prevention": (
            "Delay pruning to dry weather, make clean cuts, remove prunings from the vineyard, "
            "avoid vine stress and replant with certified clean material."
        ),
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "root_cause": (
            "Fungal infection by Pseudocercospora (Isariopsis) vitis, promoted by warm, humid "
            "conditions and dense canopies that stay wet for long periods."
        ),
        "treatment": (
            "Remove heavily affected leaves, open up the canopy and, if the disease is spreading "
            "early in the season, apply a registered protectant fungicide such as mancozeb. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Trim excess foliage for light and airflow, avoid overhead irrigation and remove fallen "
            "leaves at the end of the season."
        ),
    },
    "Grape___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Maintain canopy management and a routine protectant programme against downy and "
            "powdery mildew, and scout vines weekly through the wet season."
        ),
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "root_cause": (
            "Bacterial infection by Candidatus Liberibacter asiaticus, transmitted by the Asian "
            "citrus psyllid. Symptoms include blotchy mottle, yellow shoots and misshapen fruit. "
            "There is no cure."
        ),
        "treatment": (
            "Remove and destroy confirmed infected trees to reduce inoculum, and manage the psyllid "
            "vector with monitoring and registered insecticides in coordination with local "
            "regulators. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Plant certified disease-free nursery stock, control psyllids on flush growth, use "
            "yellow sticky monitoring and keep area-wide coordination with neighbouring orchards."
        ),
    },
    "Peach___Bacterial_spot": {
        "root_cause": (
            "Bacterial infection by Xanthomonas arboricola pv. pruni, spread by splashing rain and "
            "entering through stomata and wounds on leaves, twigs and fruit."
        ),
        "treatment": (
            "Prune out infected shoots, avoid overhead irrigation and apply copper-based sprays from "
            "leaf fall to early spring, plus during wet spells, following local label guidance. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Plant resistant cultivars, space trees for quick drying, avoid working in wet foliage "
            "and maintain balanced nitrogen to avoid soft, susceptible growth."
        ),
    },
    "Peach___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Prune annually for an open canopy, remove mummified fruit, monitor for aphids and "
            "oriental fruit moth and keep balanced nitrogen fertility."
        ),
    },
    "Pepper,_bell___Bacterial_spot": {
        "root_cause": (
            "Bacterial infection by Xanthomonas species carried in seed and crop debris, spread by "
            "rain splash and handling of wet plants."
        ),
        "treatment": (
            "Remove badly affected leaves and plants, avoid overhead irrigation and handling wet "
            "foliage, and apply copper plus mancozeb according to the label where permitted. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Use certified pathogen-free seed and transplants, rotate away from peppers and "
            "tomatoes for two years, mulch to reduce splash and disinfect tools between plants."
        ),
    },
    "Pepper,_bell___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Rotate beds, use drip irrigation, stake for airflow and scout weekly for aphids, "
            "thrips and bacterial spot after rain events."
        ),
    },
    "Potato___Early_blight": {
        "root_cause": (
            "Fungal infection by Alternaria solani. It starts on lower, older leaves and is worse "
            "under alternating wet and dry weather, low nitrogen and heavy fruit set."
        ),
        "treatment": (
            "Remove affected lower leaves, avoid overhead irrigation, maintain nitrogen fertility "
            "and apply a registered chlorothalonil, mancozeb or azoxystrobin fungicide from first "
            "symptoms. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Rotate out of potatoes and tomatoes for at least two years, use certified seed tubers, "
            "irrigate early in the day and destroy volunteer plants and cull piles."
        ),
    },
    "Potato___Late_blight": {
        "root_cause": (
            "Infection by the oomycete Phytophthora infestans. Cool, wet weather and overhead "
            "irrigation allow rapid spread; lesions expand into dark, water-soaked areas with pale "
            "sporulation on the underside."
        ),
        "treatment": (
            "Act immediately: destroy severely infected plants and cull piles, stop overhead "
            "irrigation and apply a late-blight-specific protectant (chlorothalonil or mancozeb "
            "families) on a short interval during wet weather. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Use certified seed, avoid overhead irrigation, keep cull piles destroyed, follow "
            "regional late-blight forecasting and plant less susceptible varieties where available."
        ),
    },
    "Potato___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Use certified seed tubers, rotate with non-solanaceous crops, hill soil for tuber "
            "protection and monitor leaves during cool, wet spells for blight."
        ),
    },
    "Raspberry___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Remove spent floricanes after harvest, keep rows narrow and weed-free, use drip "
            "irrigation and scout for aphids and spider mites on leaf undersides."
        ),
    },
    "Soybean___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Rotate with cereals, seed at recommended dates, keep balanced fertility and scout for "
            "soybean aphids, stink bugs and foliar spotting diseases through pod fill."
        ),
    },
    "Squash___Powdery_mildew": {
        "root_cause": (
            "Fungal infection by Podosphaera xanthii, favoured by warm days, cool nights and "
            "moderate humidity; dense canopies and excess nitrogen increase severity."
        ),
        "treatment": (
            "Remove severely affected leaves, improve airflow and use a registered sulphur, "
            "potassium bicarbonate or biofungicide product early, rotating modes of action. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Choose resistant varieties, widen plant spacing, avoid overhead irrigation and "
            "excessive nitrogen, and start scouting when nights turn cool."
        ),
    },
    "Strawberry___Leaf_scorch": {
        "root_cause": (
            "Fungal infection by Diplocarpon earlianum. Overwinters in infected leaves and spreads "
            "during wet spring weather causing purple blotches that merge as leaves scorch."
        ),
        "treatment": (
            "Remove and destroy infected leaves after harvest (bed renovation), thin plants for "
            "airflow and apply a registered captan fungicide during wet periods. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Renovate beds annually, use clean transplants, avoid overhead irrigation and reduce "
            "plant density so foliage dries quickly."
        ),
    },
    "Strawberry___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Renovate beds after harvest, maintain straw mulch, use drip irrigation and scout for "
            "mites, aphids and leaf spot on a weekly basis."
        ),
    },
    "Tomato___Bacterial_spot": {
        "root_cause": (
            "Bacterial infection by Xanthomonas species, seed-borne and surviving in debris; "
            "splashing rain and handling wet plants spread it quickly."
        ),
        "treatment": (
            "Remove infected leaves and badly infected plants, switch to drip irrigation, avoid "
            "working among wet plants and apply copper plus mancozeb where registered. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Use certified clean seed and transplants, rotate away from tomato, pepper and potato "
            "for at least two years, mulch beds and disinfect pruning tools."
        ),
    },
    "Tomato___Early_blight": {
        "root_cause": (
            "Fungal infection by Alternaria solani. It survives in debris and soil, attacks older "
            "lower leaves first and is more severe in warm, humid weather with wet-dry cycles."
        ),
        "treatment": (
            "Remove affected lower leaves, mulch to prevent soil splash, stake and prune for "
            "airflow and apply a registered chlorothalonil or mancozeb fungicide on a 7-10 day "
            "interval in humid weather. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Rotate crops for two to three years, use certified seed, irrigate at the base early in "
            "the day and avoid overcrowding plants."
        ),
    },
    "Tomato___Late_blight": {
        "root_cause": (
            "Infection by Phytophthora infestans. Cool, wet weather and leaf wetness let it spread "
            "very fast, producing greasy grey-green lesions with white sporulation underneath."
        ),
        "treatment": (
            "Remove and destroy infected plants promptly (do not compost), avoid overhead "
            "irrigation and protect remaining plants with a late-blight-specific fungicide during "
            "cool, wet weather. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Plant certified transplants, keep tomato and potato plantings apart, irrigate at the "
            "soil line, follow local blight forecasts and destroy volunteer plants."
        ),
    },
    "Tomato___Leaf_Mold": {
        "root_cause": (
            "Fungal infection by Passalora fulva, a greenhouse disease that needs very high "
            "humidity (above about 85%) and poor air movement to establish."
        ),
        "treatment": (
            "Increase ventilation, reduce humidity by heating/venting and avoid overhead watering; "
            "remove affected leaves and use a registered fungicide if the outbreak continues. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Keep night humidity below the condensation point, prune lower leaves, space plants and "
            "grow resistant greenhouse cultivars where available."
        ),
    },
    "Tomato___Septoria_leaf_spot": {
        "root_cause": (
            "Fungal infection by Septoria lycopersici. The fungus overwinters on debris and weed "
            "hosts and spreads by rain splash to lower leaves."
        ),
        "treatment": (
            "Remove infected lower leaves, mulch the soil surface, stake plants and apply a "
            "registered chlorothalonil or mancozeb fungicide at 7-10 day intervals while rain "
            "continues. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Rotate with non-solanaceous crops, remove old plant debris, control nightshade weeds "
            "around the field and water at the base."
        ),
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "root_cause": (
            "Feeding damage by Tetranychus urticae mites, which thrive in hot, dry conditions and "
            "feed on leaf undersides, producing stippling and fine webbing."
        ),
        "treatment": (
            "Remove badly infested leaves, manage plant water stress, and apply insecticidal soap "
            "or horticultural oil with good underside coverage; rotate miticide modes of action if "
            "a specific product is needed. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Maintain adequate irrigation, keep dust down along field edges, avoid broad-spectrum "
            "insecticides that kill natural enemies and monitor leaf undersides weekly."
        ),
    },
    "Tomato___Target_Spot": {
        "root_cause": (
            "Fungal infection by Corynespora cassiicola, favoured by high humidity, prolonged leaf "
            "wetness and dense canopies."
        ),
        "treatment": (
            "Improve airflow through pruning and stake, avoid overhead irrigation, remove affected "
            "leaves and apply a registered fungicide from the strobilurin or triazole families when "
            "lesions are expanding. " + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Rotate crops, destroy crop debris, avoid overcrowding and keep nitrogen balanced so "
            "the canopy stays open."
        ),
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "root_cause": (
            "Viral infection by tomato yellow leaf curl virus, transmitted by whiteflies (Bemisia "
            "tabaci). Infected plants show upward leaf curling, yellow margins and stunted growth. "
            "There is no cure."
        ),
        "treatment": (
            "Remove and destroy infected plants to reduce inoculum and manage whitefly populations "
            "with monitoring, reflective mulches and registered insecticides where necessary. "
            + CHEMICAL_APPLICATION_NOTE
        ),
        "prevention": (
            "Use TYLCV-resistant varieties, raise transplants under insect-proof screens, use "
            "yellow sticky traps and keep weed hosts around the field managed."
        ),
    },
    "Tomato___Tomato_mosaic_virus": {
        "root_cause": (
            "Viral infection by tomato mosaic virus, spread mainly by mechanical contact through "
            "hands, tools and contaminated seed, and by tobacco products. There is no cure."
        ),
        "treatment": (
            "Remove and destroy infected plants, disinfect hands and tools (soap, then a "
            "disinfectant) between plants, and avoid tobacco use near the crop."
        ),
        "prevention": (
            "Sow certified virus-free seed, control volunteer tomato and solanaceous weeds, and "
            "use resistant cultivars where available."
        ),
    },
    "Tomato___healthy": {
        "root_cause": "No disease or pest damage is visible on the analysed leaf.",
        "treatment": "No curative action needed for this leaf.",
        "prevention": (
            "Rotate beds, stake for airflow, mulch to limit soil splash, water at the base and "
            "scout weekly for early blight, whitefly and spider mites."
        ),
    },
}

UNSUPPORTED_REPORT = {
    "root_cause": (
        "The classifier did not return a label within its supported set of 38 plant disease "
        "classes, so no diagnosis can be made from this image."
    ),
    "treatment": (
        "No treatment recommendation is generated for an unsupported result. Retake the photo as a "
        "close-up of a single leaf in daylight, or consult your local extension service for a "
        "field diagnosis."
    ),
    "prevention": (
        "Continue routine scouting and record which crops and symptoms you observe so an "
        "agronomist can review them."
    ),
}

UNCERTAIN_REPORT = {
    "root_cause": (
        "The image does not match the training conditions well enough for a reliable diagnosis: "
        "the model's prediction was low-confidence or changed under small image variations."
    ),
    "treatment": (
        "No disease is claimed and no treatment is recommended from this image. Retake the photo "
        "as a sharp, evenly lit close-up of a single leaf against a plain background, or consult "
        "your local extension service."
    ),
    "prevention": (
        "Scout the crop directly and submit a clear leaf image; do not apply plant protection "
        "products without a confirmed diagnosis."
    ),
}


def split_class_label(class_name):
    """Split a PlantVillage label into (crop, condition); returns (None, class_name) if unknown."""
    if "___" not in class_name:
        return None, class_name
    crop, condition = class_name.split("___", 1)
    return crop, condition


def get_crop_display_name(class_name):
    crop, _ = split_class_label(class_name)
    if crop is None or crop not in CROP_DISPLAY_NAMES:
        return None
    return CROP_DISPLAY_NAMES[crop]


def get_condition_display_name(class_name):
    _, condition = split_class_label(class_name)
    if condition in CONDITION_DISPLAY_NAMES:
        return CONDITION_DISPLAY_NAMES[condition]
    return condition.replace("_", " ").strip()


def is_healthy_class(class_name):
    _, condition = split_class_label(class_name)
    return isinstance(condition, str) and condition.strip().lower() == "healthy"


def is_supported_class(class_name):
    return class_name in TREATMENTS
