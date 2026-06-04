import cv2
import sys
import os
import torch
import open_clip
from PIL import Image

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

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from tracker import TrackStateManager

def visualize_video(video_path, output_path="output_staff_viz.mp4"):
    print(f"Opening video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Video writer setup
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize OpenCLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP model on {device}...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text_prompts = _OCLIP_CUSTOMER_PROMPTS + _OCLIP_STAFF_PROMPTS
    text_tokens = clip_tokenizer(text_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    staff_prob_accumulator = {}
    behavior_cache = {} # tid -> {'first_frame': frame_num, 'interaction_frames': 0}

    # Construct absolute path to yolov8n.pt which is in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "yolov8n.pt")
    
    model = YOLO(model_path)
    tracker_manager = TrackStateManager()
    
    frame_num = 0
    print("Processing frames... Press 'q' to stop early if running with UI.")
    
    # Create a resizable window so the OS doesn't crop the edges of a 1080p video
    cv2.namedWindow("Staff vs Customer Visualization", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        
        # Track persons
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_detections = {}
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid_tensor in zip(r.boxes, r.boxes.id):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    tid = int(tid_tensor.item())
                    
                    if cls_id == 0 and conf > 0.3:  # class 0 = person
                        current_detections[tid] = {
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": conf,
                            "center": ((box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2, (box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2)
                        }
                        
        active_states, _, _ = tracker_manager.update(current_detections, frame_num)
        
        # --- BEHAVIOR: Proximity / Assisting check ---
        # Calculate distances between all active tracks to see who is interacting
        for state_a in active_states:
            tid_a = int(state_a.track_id.split('_')[1])
            det_a = current_detections.get(tid_a)
            if not det_a: continue
            
            if tid_a not in behavior_cache:
                behavior_cache[tid_a] = {'first_frame': frame_num, 'interaction_frames': 0}
            
            is_interacting = False
            for state_b in active_states:
                if state_a.track_id == state_b.track_id: continue
                tid_b = int(state_b.track_id.split('_')[1])
                det_b = current_detections.get(tid_b)
                if not det_b: continue
                
                # Calculate distance between centers
                dist = ((det_a["center"][0] - det_b["center"][0])**2 + (det_a["center"][1] - det_b["center"][1])**2)**0.5
                if dist < 120: # Roughly within 1-2 meters depending on camera angle
                    is_interacting = True
                    break
                    
            if is_interacting:
                behavior_cache[tid_a]['interaction_frames'] += 1
        
        # Draw bounding boxes and labels
        for state in active_states:
            tid_int = int(state.track_id.split('_')[1])
            det = current_detections.get(tid_int)
            if not det:
                continue
                
            x1, y1, x2, y2 = map(int, det["bbox"])
            
            # Extract and classify with OpenCLIP (max 10 frames to save compute)
            probs = staff_prob_accumulator.setdefault(tid_int, [])
            if len(probs) < 10:
                cy1, cy2 = max(0, y1), min(frame.shape[0], y2)
                cx1, cx2 = max(0, x1), min(frame.shape[1], x2)
                
                if cy2 > cy1 and cx2 > cx1:
                    crop = frame[cy1:cy2, cx1:cx2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    # Pad to square so CLIP's CenterCrop doesn't cut off the head/legs
                    h, w, _ = crop_rgb.shape
                    size = max(h, w)
                    pad_h = (size - h) // 2
                    pad_w = (size - w) // 2
                    padded_crop = cv2.copyMakeBorder(crop_rgb, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    
                    pil_img = Image.fromarray(padded_crop)
                    
                    image_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        # Sum the probabilities for all staff prompts
                        staff_prob = similarity[0, len(_OCLIP_CUSTOMER_PROMPTS):].sum().item()
                        probs.append(staff_prob)
            
            # Determine staff status based on average probability
            avg_prob = sum(probs) / len(probs) if probs else 0
            
            # --- HEURISTIC 2: Torso Blackness ---
            # Extract torso region (top 20% to 60% of bounding box, inner 60% width)
            t_y1, t_y2 = int(y1 + (y2-y1)*0.2), int(y1 + (y2-y1)*0.6)
            t_x1, t_x2 = int(x1 + (x2-x1)*0.2), int(x1 + (x2-x1)*0.8)
            torso_crop = frame[max(0, t_y1):min(frame.shape[0], t_y2), max(0, t_x1):min(frame.shape[1], t_x2)]
            
            if torso_crop.size > 0:
                hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                # Black color range in HSV: Value channel < 60
                import numpy as np
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, 60])
                mask = cv2.inRange(hsv, lower_black, upper_black)
                black_ratio = cv2.countNonZero(mask) / (mask.size + 1e-6)
            else:
                black_ratio = 0

            # --- BEHAVIORAL SCORING ---
            behaviors = behavior_cache.get(tid_int, {'first_frame': frame_num, 'interaction_frames': 0})
            frames_tracked = frame_num - behaviors['first_frame']
            interaction_ratio = behaviors['interaction_frames'] / max(1, frames_tracked)
            
            # Combine OpenCLIP, Torso Color, and Behavior heuristic
            # If someone spends a lot of time interacting (assisting) AND matches the color heuristic, they are staff.
            # If they have been tracked for a massive amount of time (persistence), they are likely staff.
            
            looks_like_staff = (avg_prob > 0.40) and (black_ratio > 0.10)
            acts_like_staff = interaction_ratio > 0.30 or frames_tracked > 300 # 300 frames = 20 seconds of continuous presence
            
            is_staff = looks_like_staff and acts_like_staff
            
            state.is_staff = is_staff # Update state in tracker for completeness
            
            if is_staff:
                color = (0, 0, 255) # Red for staff
                label = f"STAFF {state.track_id} (Int:{interaction_ratio:.1f})"
            else:
                color = (0, 255, 0) # Green for customer
                label = f"CUST {state.track_id} (Int:{interaction_ratio:.1f})"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        out.write(frame)
        
        # Show frame if running locally with UI
        cv2.imshow("Staff vs Customer Visualization", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    video_to_test = r"E:\purplle_tech_challenge\data\Store_1\CAM_2_zone.mp4"
    visualize_video(video_to_test)
