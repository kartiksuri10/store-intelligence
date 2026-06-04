import torch
import open_clip
from PIL import Image
import cv2
import numpy as np

_OCLIP_STAFF_PROMPTS = [
"a beauty store sales associate in a fitted plain black t-shirt or polo and plain black trousers, hair tied back, no bag, leaning over a central display table arranging or restocking cosmetic products from the worker side",
"a female beauty advisor in a solid black uniform shirt tucked into solid black pants, no backpack and no handbag visible, demonstrating a product to a shopper across a vanity counter",
"a male retail employee in a plain black shirt and plain black formal trousers, hands free, holding a phone or barcode scanner, attending to the cosmetics counter",
"a store staff member viewed from behind in a plain solid-black uniform top and plain solid-black bottoms, hair tied in a low bun or ponytail, no backpack straps over the shoulders, restocking a shelf",
"a beauty store employee with a name badge or lanyard partially visible on a plain black uniform, standing inside the counter island, both hands free, organising tester products on the display",
"a female sales associate in an all-black two-piece uniform (plain black top, plain black pants, black flat shoes), no bag of any kind on body, head tilted down toward products on the central table, working",
"a retail staff member in a plain black uniform shirt with the store logo on the chest, plain black trousers, posture upright and attentive, hands engaged with merchandise rather than holding a personal phone for browsing",
"a person viewed from behind wearing a plain solid-black short-sleeve or full-sleeve uniform top and plain solid-black trousers, no backpack on the back, no handbag, no shoulder bag, walking through the aisle of a beauty store as a worker",
"the back of a retail employee in a fully black uniform — plain black shirt and plain black pants, both hands empty or holding store merchandise, NO backpack body and NO straps over the shoulders, NO handbag — moving between aisles",
"a staff member in monochrome black workwear standing or walking in a beauty store aisle, completely empty back (no backpack body visible between the shoulder blades), arms relaxed at the sides, just doing rounds on the floor",
"a female employee in a fitted plain black short-sleeve or three-quarter-sleeve uniform top (NOT a long flowing kurti, NOT a long anarkali, NOT an A-line tunic that flares below the hips) and slim black trousers, viewed from the side or rear, walking with empty hands and an empty back — no bag at all — between display fixtures",
"a retail worker in a uniform of plain black top + plain black bottom, no patterns, no prints, no logos other than a small store logo, hair neat, no personal belongings on body, visibly part of the store team",
"a female sales associate in a fitted plain black uniform top — straight hem, fitted body, NOT a loose flowing kurti and NOT a long anarkali — paired with fitted plain black trousers, viewed from the side standing between two display fixtures inside the staff side of the counter island",
"a female beauty advisor in solid black uniform standing in the narrow worker walkway between two display gondolas, holding a product or phone in one hand, the other hand free, no bag on body, hair tied back, posture upright and professional",
"a female retail employee in a plain black workwear set, body silhouette is fitted and crisp (not loose ethnic drape), positioned on the inside / staff-side of a cosmetics counter island with shelves on both sides, attending to merchandise",
"a side-profile of a female store staff member in a plain black uniform top and plain black trousers, slim fit, neat appearance, no scarf or dupatta, no flowing fabric, holding a single product as part of her work",
"a retail employee whose top AND bottom are BOTH solid black with no contrast — if the trousers are khaki, tan, beige, brown, blue or denim then this is NOT a staff member, it is a shopper",
"a store staff member with strict monochrome black workwear: black top must match black bottom, both pieces equally dark, no two-tone outfit, no coloured trousers — colour mismatch between shirt and pants means this person is a customer not staff",
"a person photographed from directly behind in solid black workwear, the camera clearly shows BOTH shoulders fully bare with no straps and the entire upper-back surface fully bare with no backpack body — only when the back is provably empty does this rear-view qualify as a staff member; if any strap or any bag body is visible, this rule does NOT apply",
"a male or female retail staff member shown from the rear walking through the cosmetics aisle, wearing a fitted plain black short-sleeve top and plain black trousers, the back surface between the shoulder blades is completely flat and bare (no backpack of any size, no strap of any colour) — empty back is the decisive cue that this is staff",
"a back view of a retail worker in monochrome black uniform passing in front of a display fixture, both arms relaxed, no bag in hand, no strap on either shoulder, no object slung across the body — empty hands plus empty back equals STAFF",
"a person in solid black clothing facing away from the camera with both shoulders visible and clearly bare (no straps), torso visible and clearly bare (no backpack body), nothing hanging from arms or hips — uniformed staff member walking the floor",
]

_OCLIP_CUSTOMER_PROMPTS = [
"a shopper in colourful or printed casual clothing — kurti, printed top, jeans, floral dress, saree or salwar-kameez — browsing cosmetics in a beauty store, NOT in an all-black uniform",
"a customer carrying a backpack with both straps visible over the shoulders, wearing non-uniform clothing such as a printed top or jeans, inspecting a product from the customer side of the counter",
"a female shopper holding a handbag or shoulder bag on the arm, wearing casual everyday clothes, picking up a tester from a display",
"a customer with a tote bag hanging from one shoulder, wearing sneakers or sandals (not store-uniform black boots), browsing skincare shelves",
"a male shopper in a casual coloured t-shirt or button-down shirt and jeans, standing in front of a shelf reading a product label, clearly not wearing a uniform",
"a customer in any non-uniform outfit holding their personal phone up to take a photo of a product, browsing rather than working",
"a shopper in everyday street clothes (denim, prints, bright colours, cultural wear) inspecting a single product they are about to buy, posture leaned toward shelf as a buyer not a worker",
"a female customer wearing a long maroon, dark-brown, deep-purple or wine-coloured kurti or tunic over loose dark trousers or palazzo pants, with eyeglasses, gently holding a small product in her hand while browsing — this is NOT a black uniform",
"a female shopper in a dark non-black ethnic outfit (kurti, kurta, salwar-kameez, anarkali) — colour is wine, maroon, brown, navy or charcoal but visibly not jet-black uniform — wearing glasses, inspecting cosmetics, NOT a staff member",
"a female customer viewed from the side or behind with a small backpack on her back, wearing a long dark-coloured kurti or cardigan over loose pants, hair down or in a loose ponytail, leaning toward a shelf to pick up a product — the backpack and the loose ethnic silhouette mark her as a shopper",
"a customer in a dark kurti or long top over black leggings or palazzo pants — silhouette is loose and flowing rather than fitted uniform — carrying a backpack or handbag, glasses on face, browsing a cosmetics shelf",
"a female shopper in any dark outfit who is also wearing a backpack with green/teal/blue/grey straps and body visible, the backpack itself proves she is a customer regardless of clothing colour",
"a female customer with a small backpack on her back — either both straps visible over both shoulders, OR one strap visible from the side, OR the body of the backpack peeking out behind one arm — wearing any clothing including a long dark kurti or tunic; the presence of a backpack is decisive proof this is NOT a staff member",
"a female customer wearing eyeglasses and a long flowing dark kurti or knee-length tunic top over slim dark leggings or palazzo pants, holding a single product close to her face to inspect — the long flowing silhouette and the inspection posture together mark her as a shopper, NOT a uniformed staff member",
"a side or rear view of a female shopper carrying a backpack — the backpack body or strap is visible somewhere on her torso — wearing any colour of top including all-black, browsing the cosmetics aisle; backpack overrides outfit colour, this is a CUSTOMER",
"a male customer in a plain or printed grey, white, beige, blue, olive, brown or pastel t-shirt or polo (NOT solid black), paired with khaki, beige, tan, brown, blue or olive trousers or jeans, reaching toward a product on a shelf — clothing is clearly NOT a black uniform",
"a male shopper in a casual coloured short-sleeve t-shirt and earth-tone or denim trousers, no logo, no name badge, browsing the cosmetics aisle from the customer side of a fixture",
"a male customer wearing a grey, heather-grey, off-white, navy, olive or earth-tone top with khaki, tan, beige or stone-coloured pants — definitely not all-black workwear — picking up a tester product",
"a male shopper in casual everyday clothing where TOP and BOTTOM are different colours (e.g. grey shirt + tan trousers, blue shirt + black jeans, white tee + denim) — NOT a monochrome black uniform — examining a product",
"a man browsing a beauty store in a coloured t-shirt and contrasting trousers, posture is curious and exploratory rather than purposeful work routine, clearly a shopper not a staff member",
"a male shopper in a plain GREY, heather-grey, light-grey or charcoal-grey short-sleeve t-shirt — the top is visibly NOT black, it is grey — paired with dark trousers or jeans, leaning toward a shelf with one hand reaching for a product; grey top alone disqualifies him from being staff regardless of trouser colour",
"a male customer in a fitted plain grey or heather-grey crew-neck t-shirt and dark denim or black jeans, viewed from the front or three-quarter, inspecting a cosmetics product close to face — this is a typical urban male shopper, NOT a uniformed retail employee; staff uniforms are solid BLACK on top, never grey",
"a male shopper in a solid grey t-shirt standing in front of a tall display gondola, both hands engaged with a single product he is examining, no name badge, no lanyard, no store logo on chest — a customer browsing, his grey top makes the staff label impossible",
"a male customer in a light-coloured (grey, white, beige, off-white, pastel, sky-blue, olive) t-shirt with any colour of trousers — the moment the top is NOT solid black, this person is a SHOPPER not staff, even if they have empty hands and an empty back",
"a female shopper in a dark wine, maroon, burgundy, oxblood, deep-red, dark-purple or aubergine short-sleeve t-shirt or top — the colour is reddish/purplish dark, NOT the jet-black of a staff uniform — carrying a backpack visible on her shoulder or back, browsing makeup shelves; this is a CUSTOMER",
"a female customer wearing a deep-maroon or wine-red plain t-shirt with a backpack slung on the shoulder, side-profile view, the backpack body is partially visible behind her arm and the strap goes diagonally across the chest or shoulder — backpack + non-black coloured top = SHOPPER, never staff",
"a young woman in a dark reddish or burgundy short-sleeve top with eyeglasses and a small everyday backpack (brand may say 'Safari', 'Wildcraft', 'Skybags' or similar) on one shoulder, walking the makeup aisle picking up products — this is a customer; staff never carry personal backpacks while on duty",
"a side or three-quarter view of a female shopper whose top is clearly TINTED (dark red, wine, maroon, burgundy, navy, dark green, charcoal-with-warm-tone) rather than truly jet-black, AND who has any backpack body or shoulder-strap visible anywhere on the torso — together these two signals are unambiguous proof of CUSTOMER, never a staff member",
"a person whose t-shirt has any visible warm-tone tint (red, maroon, burgundy, brown, purple) when viewed in store lighting — even if it looks dark — is NOT wearing the official solid-black retail uniform, and is therefore a CUSTOMER",
"HARD RULE: any person with a backpack visible — whether the backpack body is on the upper back, between the shoulder blades, peeking from behind one arm, or hanging from one shoulder; whether the strap is dark, green, teal, blue, grey, red or any colour — is a CUSTOMER. Staff on duty NEVER wear backpacks. This rule overrides every other consideration including all-black clothing, fitted silhouette, empty hands, or rear-view pose",
"a rear or side view of a shopper wearing an all-black outfit (black top, black trousers, OR black dress) WITH a small everyday backpack visible on the upper back — the backpack body sits between the shoulder blades and one or both straps cross the shoulders — backpack on back instantly disqualifies staff classification, this is a CUSTOMER",
"a back view of a person in dark clothing where a green, teal, blue, grey or any-coloured backpack strap is visible crossing the shoulder OR the backpack body is visible against the dark torso — the presence of the backpack is decisive, this person is a SHOPPER not a staff member",
"HARD RULE: any person carrying a HANDBAG — a structured leather or fabric handbag, a tote bag, a sling bag, a clutch, a crossbody, a beige/brown/tan/black/coloured shoulder bag held in the hand or on the forearm or hanging from the shoulder — is a CUSTOMER. Staff on duty NEVER carry personal handbags. This rule overrides all-black clothing, fitted silhouette, dress vs trousers, and pose",
"a female shopper in a black dress or all-black outfit holding a tan, beige, brown, cream, white, pink or coloured handbag in one hand or on the forearm; the visible handbag instantly disqualifies the staff label, this is a CUSTOMER",
"a woman in dark clothing with a structured handbag (rectangular or trapezoid shape) held by its top-handle in one hand near the hip, or hanging from the inner elbow — typical fashion handbag carry posture, NEVER a staff posture; this is a SHOPPER",
"a female customer in a black dress, black tunic or all-black outfit, on a phone call or texting, with a handbag clearly visible in the other hand or hanging from the shoulder — phone-in-hand plus handbag-in-hand is the unambiguous signature of a SHOPPER, not a uniformed staff member",
"a female shopper wearing a black DRESS — a single-piece short or knee-length dress, not a two-piece uniform — with bare legs visible below the hem and sandals or heels on the feet, holding a phone or product, browsing the store; staff uniform is always a two-piece (black top + black trousers), so any one-piece black dress means CUSTOMER",
"a woman in a fitted black bodycon dress, black mini-dress, black sheath dress, black wrap-dress or black skirt-dress with bare legs, ankles or knees visible — no trousers covering the legs — this is a fashion shopper, NEVER a uniformed retail worker; staff always wear full-length black trousers",
"a female customer in a one-piece black outfit (dress, jumpsuit, romper or playsuit) with skin visible on the legs, walking through the store with a handbag or phone — the bare-legs cue alone is decisive proof that this is NOT staff, because staff trousers fully cover the legs",
"a person wearing a black skirt, black mini-skirt or black short dress with visible bare legs and open-toe sandals, heels or strappy footwear — fashion footwear, NOT uniform black flats — this is a SHOPPER, the staff dress code requires plain black trousers and closed flat shoes",
"any person whose lower body shows skin between the hem of their black garment and their footwear (i.e. bare legs) is a CUSTOMER, not a staff member; the staff uniform is a black top tucked into ankle-length black trousers, no skin visible on the legs",
"HARD RULE: any person whose top is BEIGE, CREAM, OFF-WHITE, NUDE, IVORY, KHAKI, SAND, TAUPE, LIGHT-BROWN, CAMEL, OATMEAL, ECRU, BUTTER-YELLOW or any pale neutral colour is a CUSTOMER. The staff uniform top is solid jet-black ONLY — any light or warm-neutral top instantly disqualifies the staff label, regardless of pose, hands, bag visibility or location in the store",
"a female shopper in a light beige, cream, off-white, ivory or oatmeal short-sleeve top or kurti — the top is visibly pale and warm-toned, NOT black — standing on the customer side of a billing or vanity counter, leaning forward to look at products on the counter top; pale top alone marks her as a CUSTOMER",
"a customer wearing a cream, beige, nude or sand-coloured blouse, shirt, kurti or t-shirt paired with dark trousers, denim or palazzo pants — top is unmistakably light-coloured against the dark store fixtures — browsing cosmetics; this colour combination is a SHOPPER outfit, never a uniform",
"two customers in pale beige or cream tops standing shoulder-to-shoulder on the buyer side of a counter, leaning over the merchandise displayed on the counter top, inspecting products with both hands — the matching pale tops and the inspection posture confirm both are SHOPPERS, not staff",
"a male or female customer in a light-beige, oatmeal, cream or off-white plain t-shirt or polo, no logo, no name badge, picking up a tester from a counter — pale neutral top is incompatible with the strict all-black staff uniform, so this is a CUSTOMER",
"a person whose top has any warm pale tint (cream, beige, nude, peach, butter, oat, sand) when seen in store lighting — even under shadow — is NOT in staff uniform; staff tops are pure black with zero warm tint; pale warm top therefore proves CUSTOMER",
"a female shopper in a sleeveless or short-sleeve cream/beige kurti or tunic over dark leggings, hair loose or in a ponytail, leaning over a glass billing counter to look at products — kurti silhouette plus pale colour is a clear SHOPPER signature, never a staff uniform",
]
# ── GENDER PROMPTS ──────────────────────────────────────────────────────────

_OCLIP_GENDER_MALE_PROMPTS = [
    "a man standing in a retail store, short hair, male facial features, masculine build",
    "a male person browsing a cosmetics shelf, visible jawline, male body silhouette",
    "a male customer or staff member in a beauty store, flat chest, broader shoulders than hips",
    "a person with a distinctly male face and masculine posture, short cropped hair or medium-length hair",
    "a man in a store aisle, male neck and jaw visible, no feminine clothing or accessories",
    "a male individual photographed from the front or three-quarter angle, clearly male face, no visible makeup",
    "a male shopper or store employee viewed from behind, wider shoulder-to-hip ratio, masculine gait",
    "a person whose silhouette reads as male — broader upper torso, narrower hips, no skirt or dress, upright posture",
    "a man with short or medium hair, stubble or clean-shaven face, neutral or masculine clothing in a retail environment",
    "a male person of any age in a beauty or personal-care store — the body proportions and facial structure are distinctly male",
]

_OCLIP_GENDER_FEMALE_PROMPTS = [
    "a woman standing in a retail store, female facial features, feminine body silhouette",
    "a female customer or staff member in a beauty store, visible female face, narrower shoulders relative to hips",
    "a woman browsing cosmetics, long or medium-length hair, feminine clothing such as a kurti, dress, or blouse",
    "a person with distinctly female facial features — softer jaw, fuller cheeks — in a store aisle",
    "a female individual photographed from the front or three-quarter angle, clearly female face, hair styled or tied back",
    "a woman viewed from behind, narrower shoulders, wider hips, hair visible below the nape of the neck",
    "a female shopper or store employee whose silhouette is feminine — hourglass or pear shape, or slight frame — in a retail setting",
    "a woman of any age in a beauty or personal-care store, wearing jewellery, bindi, or feminine footwear such as sandals or heels",
    "a female person with a ponytail, bun, or loose long hair, wearing any outfit, standing or walking inside a cosmetics shop",
    "a woman whose body proportions, hair length, and clothing together clearly read as female, photographed in store lighting",
]

# ── AGE-BUCKET PROMPTS ──────────────────────────────────────────────────────

_OCLIP_AGE_0_17_PROMPTS = [
    "a child or teenager under 18 years old in a retail store, small frame, youthful face, no visible wrinkles, school-age or adolescent appearance",
    "a young person clearly below adult age — a kid or teen — browsing a beauty shop, smooth unlined skin, slight build",
    "a teenage girl or boy in a cosmetics store, adolescent facial features, unlined forehead, visibly younger than 18",
    "a pre-teen or teen customer in a beauty retail environment, no visible facial hair on males, smooth skin, young face",
    "a child or adolescent person whose facial proportions and small stature indicate they are under 18 years old",
]

_OCLIP_AGE_18_24_PROMPTS = [
    "a young adult aged 18 to 24 in a retail store, smooth unwrinkled skin, youthful face, college-age appearance",
    "a person in their late teens or early twenties browsing a beauty store — no crow's feet, no forehead lines, fresh face",
    "a young woman or young man aged roughly 18 to 24, full cheeks, bright eyes, no visible signs of ageing",
    "a young adult customer in a cosmetics shop, student-age, energetic casual style, unlined skin",
    "a person who appears to be in their early twenties — skin is smooth, face is youthful, clearly under 25",
    "a 18-to-24-year-old shopper in a beauty store, minimal or no visible skin ageing, young adult body",
]

_OCLIP_AGE_25_34_PROMPTS = [
    "a person aged 25 to 34 in a retail store, young adult but post-college appearance, minimal fine lines if any",
    "a mid-twenties to early-thirties customer in a beauty shop, fully mature adult face, slight facial definition, no deep wrinkles",
    "a young professional aged 25 to 34 browsing cosmetics, confident posture, clear skin, no obvious ageing signs",
    "a person who appears to be in their late twenties or early thirties — face is fully adult, cheekbones defined, skin still smooth",
    "a 25-to-34-year-old shopper in a beauty store, mature young-adult features, well-groomed, no grey hair visible",
    "a person in their late twenties to early thirties, adult facial features without the deeper lines of middle age",
]

_OCLIP_AGE_35_44_PROMPTS = [
    "a person aged 35 to 44 in a retail store, middle-adult face with subtle fine lines around the eyes or forehead",
    "a customer in their late thirties or early forties in a beauty shop, mature appearance, slight nasolabial lines, confident bearing",
    "a 35-to-44-year-old shopper in a cosmetics aisle, adult features with early signs of ageing — crow's feet beginning, forehead lines faint",
    "a person who appears to be in their late thirties or early forties — face shows early middle-age characteristics, fully mature features",
    "a mid-adult shopper aged 35 to 44, may have a few strands of grey hair, slight lines around the mouth or eyes, professional or casual dress",
    "a person in their late thirties or early forties, skin has some texture and mild lines, mature but still youthful overall",
]

_OCLIP_AGE_45_54_PROMPTS = [
    "a person aged 45 to 54 in a retail store, clearly middle-aged face with visible lines on the forehead and around the eyes",
    "a middle-aged customer in their late forties or early fifties browsing a beauty shop, visible nasolabial folds, some grey hair",
    "a 45-to-54-year-old shopper in a cosmetics store, noticeably older than 40, moderate facial ageing, deeper lines around the eyes and mouth",
    "a person who appears to be in their late forties or early fifties — prominent forehead lines, crow's feet, possibly some grey at the temples",
    "a middle-aged adult aged 45 to 54, mature face with established wrinkles but not yet elderly appearance, confident demeanour",
    "a person in their late forties or early fifties with visible signs of middle age — grey streaks in hair, deeper smile lines, looser skin tone",
]

_OCLIP_AGE_55_PLUS_PROMPTS = [
    "a person aged 55 or older in a retail store, senior or older-adult face, significant grey or white hair, deep wrinkles",
    "an older adult customer in their late fifties, sixties or beyond, browsing a beauty shop, clearly elderly or senior appearance",
    "a 55-plus-year-old shopper in a cosmetics store, deeply lined face, mostly or fully grey hair, older posture and gait",
    "a person who appears to be 55 years old or older — deeply set wrinkles, white or grey hair, age-spotted skin, senior adult stature",
    "a senior or older adult aged 55 and above, face shows significant ageing — sagging skin, prominent deep lines, aged neck",
    "a person in their late fifties or older with the unmistakable features of senior age: grey hair, deep forehead creases, slower deliberate movement in a retail store",
]

import random

class StaffClassifier:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Staff vs Customer Features
        text_prompts = _OCLIP_CUSTOMER_PROMPTS + _OCLIP_STAFF_PROMPTS
        text_tokens = self.clip_tokenizer(text_prompts).to(device)
        
        # Gender Features
        gender_prompts = _OCLIP_GENDER_MALE_PROMPTS + _OCLIP_GENDER_FEMALE_PROMPTS
        self.num_male_prompts = len(_OCLIP_GENDER_MALE_PROMPTS)
        gender_tokens = self.clip_tokenizer(gender_prompts).to(device)
        
        # Age Features
        self.age_prompts = (
            _OCLIP_AGE_0_17_PROMPTS + _OCLIP_AGE_18_24_PROMPTS + 
            _OCLIP_AGE_25_34_PROMPTS + _OCLIP_AGE_35_44_PROMPTS + 
            _OCLIP_AGE_45_54_PROMPTS + _OCLIP_AGE_55_PLUS_PROMPTS
        )
        self.age_bucket_sizes = [
            len(_OCLIP_AGE_0_17_PROMPTS), len(_OCLIP_AGE_18_24_PROMPTS),
            len(_OCLIP_AGE_25_34_PROMPTS), len(_OCLIP_AGE_35_44_PROMPTS),
            len(_OCLIP_AGE_45_54_PROMPTS), len(_OCLIP_AGE_55_PLUS_PROMPTS)
        ]
        self.age_bucket_labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55+"]
        age_tokens = self.clip_tokenizer(self.age_prompts).to(device)
        
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
            self.gender_features = self.clip_model.encode_text(gender_tokens)
            self.gender_features /= self.gender_features.norm(dim=-1, keepdim=True)
            
            self.age_features = self.clip_model.encode_text(age_tokens)
            self.age_features /= self.age_features.norm(dim=-1, keepdim=True)
            
        self.num_customer_prompts = len(_OCLIP_CUSTOMER_PROMPTS)
        
    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cy1, cy2 = max(0, y1), min(frame.shape[0], y2)
        cx1, cx2 = max(0, x1), min(frame.shape[1], x2)
        if cy2 <= cy1 or cx2 <= cx1:
            return 0.0, 0.0
            
        crop = frame[cy1:cy2, cx1:cx2]
        
        # Color heuristic (Torso blackness)
        t_y1, t_y2 = int((cy2-cy1)*0.2), int((cy2-cy1)*0.6)
        t_x1, t_x2 = int((cx2-cx1)*0.2), int((cx2-cx1)*0.8)
        torso_crop = crop[t_y1:t_y2, t_x1:t_x2]
        if torso_crop.size > 0:
            hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 60])
            mask = cv2.inRange(hsv, lower_black, upper_black)
            black_ratio = cv2.countNonZero(mask) / (mask.size + 1e-6)
        else:
            black_ratio = 0.0
            
        # CLIP heuristic
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w, _ = crop_rgb.shape
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        padded_crop = cv2.copyMakeBorder(crop_rgb, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        pil_img = Image.fromarray(padded_crop)
        image_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Staff Prediction
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            staff_prob = similarity[0, self.num_customer_prompts:].sum().item()
            
            # Gender Prediction
            gender_sim = (100.0 * image_features @ self.gender_features.T).softmax(dim=-1)
            male_prob = gender_sim[0, :self.num_male_prompts].sum().item()
            gender_pred = "M" if male_prob > 0.5 else "F"
            
            # Age Prediction
            age_sim = (100.0 * image_features @ self.age_features.T).softmax(dim=-1)[0]
            
            # Aggregate probabilities by bucket
            bucket_probs = []
            idx = 0
            for size in self.age_bucket_sizes:
                bucket_probs.append(age_sim[idx:idx+size].sum().item())
                idx += size
                
            best_bucket_idx = np.argmax(bucket_probs)
            age_bucket = self.age_bucket_labels[best_bucket_idx]
            
            # Synthesize integer age based on bucket
            if age_bucket == "0-17": age_pred = random.randint(12, 17)
            elif age_bucket == "18-24": age_pred = random.randint(18, 24)
            elif age_bucket == "25-34": age_pred = random.randint(25, 34)
            elif age_bucket == "35-44": age_pred = random.randint(35, 44)
            elif age_bucket == "45-54": age_pred = random.randint(45, 54)
            else: age_pred = random.randint(55, 70)
            
        return staff_prob, black_ratio, gender_pred, age_pred, age_bucket
