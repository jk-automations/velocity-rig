"""Velocity Rig library: camera rigs, worlds, light and action beats for fast car races, compiled into prompts.

Pure Python (no ComfyUI imports), so the node pack, the batch runner and tests all share it.

    shot = compose(car="a gloss teal wedge coupe #27", rival="a gloss crimson widebody fastback #14",
                   world="mountain pass", rig="auto (race sequence)", shot_index=0, intensity=0.85, seed=7)
    shot["plate_prompt"]   # first frame for Z-Image / Qwen edit, framed the way the rig sees the race
    shot["motion_prompt"]  # LTX-2.5 prompt: camera move, race action beat, speed cues and engine sound
    shot["fx"]             # Velocity FX defaults for this rig (shake, speed blur, zoom punch, chroma, ramp)

Every rig is written the way real car-commercial and motorsport crews shoot: suction-cup mounts on the car,
a Russian-arm crane on a chase car, FPV drones, trackside cameras. Cars are original designs with no real badges.
"""
from __future__ import annotations

import random

NO_BRAND = ("Original fictional race car designs with no real-world brand badges, no manufacturer emblems, no logos, "
            "no sponsor wordmarks and no readable license plates.")
PLATE_NEG = ("blurry, lowres, watermark, logo, brand emblem, text, letters, license plate text, deformed car, extra wheels, "
             "melted bodywork, people, pedestrians, faces, toy car, miniature, cartoon, CGI render, video game HUD, UI overlay, "
             "split screen, collage, multiple panels")
LTX_NEG = ("pc game, console game, video game, HUD, speedometer overlay, cartoon, childish, ugly, static camera, slow motion, "
           "still frame, frozen, text, watermark, logo, subtitles, deformed car, extra wheels, morphing car, car changes shape, "
           "people, faces, brand badges, camera pulls back, zoom out, car shrinks into the distance, car leaves frame")

# ----------------------------------------------------------------------------------------------------- animation & visual styles
STYLES = {
    "photoreal": dict(
        label="Ultra Realistic (Large-Format 70mm Cinema)",
        plate_desc="Award-winning photorealistic live-action motorsport cinematography, shot on a large-format digital cinema camera, master automotive commercial color grade, authentic optical motion blur, tack-sharp surface textures",
        motion_desc="Cinematic live-action photorealism, authentic heavy vehicle inertia and suspension physics, natural motion blur",
        plate_neg="cartoon, anime, 3d render, cgi, illustration, drawing, painting, video game HUD, lowres",
        video_neg="cartoon, anime, 3d render, cgi, illustration, drawing, video game, plastic, unrealistic physics",
        fx_tweak=dict(grain=1.0, chroma=1.0, speed_blur=1.0),
    ),
    "anime": dict(
        label="Japanese Anime (Mountain Touge)",
        plate_desc="High-octane Japanese anime aesthetic, dynamic sakuga keyframe animation, bold ink linework, vibrant multi-tone cel shading, hand-painted anime background art, dramatic manga action speed lines, classic hand-drawn mountain-pass racing anime style",
        motion_desc="Dynamic Japanese anime sakuga animation, hand-drawn anime vehicle motion, exaggerated anime speed streaks and smoke trails, dramatic anime keyframe action",
        plate_neg="photograph, photorealistic live-action, 3d cgi render, realistic human skin, live-action film",
        video_neg="photograph, photorealistic live-action, realistic 3d cgi, live-action film, realistic human skin",
        fx_tweak=dict(grain=0.2, chroma=1.6, speed_blur=1.25),
    ),
    "cartoon_3d": dict(
        label="Stylized 3D Animation (Comic-Book CG)",
        plate_desc="Stylized 3D animated feature film aesthetic, comic-book halftone CG feature style, tactile vinyl and enamel surfaces, whimsical stylized vehicle proportions, volumetric studio lighting, rich saturated color palette",
        motion_desc="Exaggerated animated feature film motion, expressive squash-and-stretch suspension physics, stylized puff tire smoke, playful high-energy pacing",
        plate_neg="photograph, live-action footage, gritty realism, grainy film, dark gritty horror",
        video_neg="photograph, live-action, gritty realism, grainy film, realistic human skin",
        fx_tweak=dict(grain=0.0, chroma=1.2, speed_blur=0.9),
    ),
    "synthwave_retro": dict(
        label="80s Retro Synthwave Night Drive",
        plate_desc="1980s retro synthwave aesthetic, vibrant neon magenta, cyan and chrome reflections, retro 80s arcade racing style, wireframe grid accents, glowing horizon sun, subtle VHS phosphor glow",
        motion_desc="High-speed retro 80s arcade racing motion, glowing neon light streaks, synthwave aesthetic, pulsing chrome reflections",
        plate_neg="dull colors, modern desaturated film, black and white, earthy tones",
        video_neg="dull colors, modern desaturated film, black and white",
        fx_tweak=dict(grain=1.3, chroma=2.0, speed_blur=1.15),
    ),
    "cyberpunk_anime": dict(
        label="Cyberpunk Anime (Neon City)",
        plate_desc="Cyberpunk anime aesthetic, neon-drenched rain reflections on wet chrome, dynamic glowing vector line accents, high-contrast dark futuristic anime",
        motion_desc="Hyper-kinetic cyberpunk anime animation, flashing neon bokeh, high-speed anime vehicle physics with electric sparks and light trails",
        plate_neg="photograph, live-action, daylight countryside, pastel colors, vintage film",
        video_neg="photograph, live-action, realistic film, daylight rural",
        fx_tweak=dict(grain=0.3, chroma=1.8, speed_blur=1.3),
    ),
    "comic_graphic_novel": dict(
        label="Comic Book Graphic Novel (Noir / Pop-Art)",
        plate_desc="Stylized graphic novel comic book illustration, heavy black ink cross-hatching, halftone Ben-Day dot shading, dramatic high-contrast noir lighting, graphic pop-art color palette",
        motion_desc="Animated graphic novel motion, dynamic comic panel physics, bold ink motion smears, stylized graphic lighting contrasts",
        plate_neg="photograph, photorealism, soft blur, 3d cgi render",
        video_neg="photograph, photorealism, 3d cgi render, smooth gradients",
        fx_tweak=dict(grain=0.5, chroma=0.8, speed_blur=0.85),
    ),
}


def build_negative_prompt(base_neg: str, style_neg: str, style: str) -> str:
    terms = [t.strip() for t in base_neg.split(",") if t.strip()]
    if style in ("anime", "cyberpunk_anime"):
        terms = [t for t in terms if not any(w in t.lower() for w in ("cartoon", "anime", "illustration", "drawing"))]
    elif style == "cartoon_3d":
        terms = [t for t in terms if not any(w in t.lower() for w in ("cartoon", "cgi", "3d"))]
    elif style == "comic_graphic_novel":
        terms = [t for t in terms if not any(w in t.lower() for w in ("cartoon", "illustration", "drawing"))]
    style_terms = [t.strip() for t in style_neg.split(",") if t.strip()]
    all_terms = terms + [st for st in style_terms if st not in terms]
    return ", ".join(all_terms)


# ----------------------------------------------------------------------------------------------------- archetypes
ARCHETYPES = {
    "car": dict(
        label="Supercars & Race Cars",
        default_subject="an original tube-frame rally-raid race buggy with an exposed bright orange roll cage, long-travel "
                        "suspension, huge knobby off-road tires on silver beadlock wheels, a short gloss teal fiberglass nose and side pods, twin roof-mounted "
                        "LED light bars and a white racing number 27 in a roundel on the side pod",
        default_rival="an original armored off-road racing truck with a flat-faced crimson red cab, black steel "
                      "bash plates, a tall snorkel, a roof rack of round spotlights, matte black wheels and a white racing number 14",
        no_brand="Original fictional race car designs with no real-world brand badges, no manufacturer emblems, no logos, "
                 "no sponsor wordmarks and no readable license plates.",
        plate_neg="blurry, lowres, watermark, logo, brand emblem, text, letters, license plate text, deformed car, extra wheels, "
                  "melted bodywork, people, pedestrians, faces, toy car, miniature, cartoon, CGI render, video game HUD, UI overlay, "
                  "split screen, collage, multiple panels",
        video_neg="pc game, console game, video game, HUD, speedometer overlay, cartoon, childish, ugly, static camera, slow motion, "
                  "still frame, frozen, text, watermark, logo, subtitles, deformed car, extra wheels, morphing car, car changes shape, "
                  "people, faces, brand badges, camera pulls back, zoom out, car shrinks into the distance, car leaves frame",
        hold_mount="The camera rig is bolted to the car, so the car's bodywork stays locked in the same place in frame the whole time. "
                   "The camera never pulls back, never zooms out and never separates from the car; the same part of the car "
                   "fills the same part of the frame from the first frame to the last.",
        hold_chase="The hero car stays large and centered in the frame the whole time, filling at least a third of the frame; "
                   "the camera keeps pace at the same distance, never falls behind, never pulls back and never loses it.",
        hold_trackside="The camera tracks the cars cleanly across the frame the whole time.",
        sequence=[
            ("wheel cam (rear)", "launch"),
            ("bumper cam", "drafting"),
            ("exhaust close-up", "hard brake"),
            ("cockpit pov", "drafting"),
            ("wheel cam (front)", "hairpin"),
            ("underbody cam", "flat out"),
            ("fpv drone dive", "drift"),
            ("door mount", "overtake"),
            ("side mirror cam", "lead defense"),
            ("rear bumper (reverse)", "lead defense"),
            ("roadside whip pan", "finish flyby"),
            ("arm car chase", "carve a bend"),
        ],
        auto_order=["arm car chase", "wheel cam (front)", "fpv drone dive", "door mount", "roadside whip pan", "bumper cam",
                    "top-down drone", "wheel cam (rear)", "arm car orbit", "crash zoom", "exhaust close-up", "fpv drone chase",
                    "low pass-by", "roof cam", "rear bumper (reverse)", "helicopter wide"],
    ),
    "motorcycle": dict(
        label="Superbikes & GP Prototypes",
        default_subject="an original aerodynamic track superbike painted gloss electric cyan and carbon black with sharp aero winglets, "
                        "gold inverted front suspension forks, titanium exhaust canister and a white racing number 64 on the front fairing, "
                        "piloted by a pro racer in a matching cyan leather racing suit, knee pucks and aerodynamic racing helmet with dark visor",
        default_rival="an original factory prototype superbike painted gloss race orange and pearl white with dual carbon aero winglets, "
                      "bronze lightweight forged wheels, titanium header pipes and bold racing number 17 on the fairing, "
                      "piloted by a competing racer in an orange-and-white leather racing suit and matching helmet",
        no_brand="Original fictional racing superbike designs with no real-world manufacturer badges, no brand logos, no sponsor wordmarks, and no readable license plates.",
        plate_neg="blurry, lowres, watermark, logo, brand emblem, text, letters, extra wheels, four wheels, car, bicycle, melted bodywork, missing limbs, deformed hands, cartoon, CGI render, video game HUD, UI overlay, split screen",
        video_neg="pc game, video game, HUD, speedometer overlay, cartoon, childish, ugly, static camera, slow motion, still frame, frozen, text, watermark, logo, deformed bike, extra wheels, bike changes shape, rider detaches from bike, camera pulls back, bike shrinks into distance",
        hold_mount="The camera is bolted directly to the motorcycle chassis, so the bike's fairing, fork, or handlebar stay anchored firmly in frame the whole time. "
                   "The camera leans and tilts with the bike in every corner, never zooming out or pulling away from the machine.",
        hold_chase="The hero superbike stays large and prominent in the center third of the frame the whole time; the chase camera matches its speed and lean angle perfectly.",
        hold_trackside="The trackside camera pans smoothly tracking the racing superbikes as they streak past inches above the asphalt.",
        sequence=[
            ("moto rear swingarm", "wheelie launch"),
            ("moto handlebar pov", "high-speed tuck"),
            ("moto knee puck cam", "knee-down carve"),
            ("moto front fork cam", "hard trail brake"),
            ("fpv drone chase", "slipstream pass"),
            ("moto tail reverse", "lead defense"),
            ("moto exhaust macro", "quickshifter pop"),
            ("roadside whip pan", "finish flyby"),
        ],
        auto_order=["moto handlebar pov", "moto knee puck cam", "moto front fork cam", "moto rear swingarm",
                    "fpv drone chase", "moto tail reverse", "moto exhaust macro", "roadside whip pan"],
    ),
    "jet": dict(
        label="Fighter Jets & Supersonic Flight",
        sound="deafening supersonic twin afterburner roar, sonic boom shockwave, jet turbine whine",
        default_subject="an original supersonic twin-engine stealth fighter jet painted matte gunmetal grey with angular radar-absorbent faceted panels, "
                        "twin canted vertical tails, glowing cyan afterburner exhaust rings, and bold tactical white insignia '01' on the tail, "
                        "piloted by a fighter pilot in high-G flight suit and dark helmet visor",
        default_rival="an original delta-wing supersonic interceptor jet painted matte arctic splinter camouflage with forward canards, "
                      "glowing bright orange dual afterburner nozzles, and tactical white insignia '07' on the twin vertical stabilizers",
        no_brand="Original fictional military aircraft designs with no real-world nation flags, no real air force roundels, no brand logos, and no readable text.",
        plate_neg="blurry, lowres, watermark, logo, text, propeller, civil airliner, passenger plane, wheels down, landed, cartoon, CGI render, video game HUD, cockpit glass reflection artifacts, split screen",
        video_neg="pc game, video game, HUD, flight sim overlay, cartoon, ugly, static camera, slow motion, still frame, frozen, text, watermark, deformed jet, wings melting, jet changes shape, camera pulls back, jet shrinks into distance",
        hold_mount="The camera is bolted to the aircraft hardpoint or fuselage, so the jet's wing, canopy frame, or tail stay locked solidly in frame the whole time while clouds, terrain, and afterburner flame rush past at supersonic speed.",
        hold_chase="The chase aircraft camera holds tight formation with the hero jet, keeping the hero jet massive and sharp in frame through high-G maneuvers.",
        hold_trackside="The ground tracking camera snaps violently tracking the supersonic fighter jet as it blasts through low altitude.",
        sequence=[
            ("jet wingtip mount", "afterburner climb"),
            ("jet canopy pov", "canyon trench run"),
            ("jet afterburner macro", "mach punch"),
            ("jet tail forward", "barrel roll evade"),
            ("formation chase", "combat formation"),
            ("jet belly pass", "supersonic flyby"),
        ],
        auto_order=["jet canopy pov", "jet wingtip mount", "jet afterburner macro", "jet tail forward", "formation chase", "jet belly pass"],
    ),
        "jet_pilot_pov": dict(
        label="Fighter Jet (Cockpit Pilot POV)",
        sound="deafening supersonic twin afterburner roar, sonic boom shockwave, jet turbine whine, cockpit heavy breathing",
        default_subject="an original supersonic twin-engine stealth fighter jet painted matte gunmetal grey with angular radar-absorbent faceted panels, glowing cyan afterburners, piloted by a focused fighter pilot in a high-G tactical flight suit and flight helmet with visor slightly raised showing their face",
        default_rival="an original delta-wing supersonic interceptor jet painted matte arctic splinter camouflage with forward canards and glowing orange afterburners",
        no_brand="Original fictional military aircraft designs with no real-world nation flags, no real air force roundels, no brand logos, and no readable text.",
        plate_neg="blurry, lowres, watermark, logo, text, propeller, civil airliner, passenger plane, wheels down, landed, cartoon, CGI render, video game HUD, split screen",
        video_neg="pc game, video game, HUD, flight sim overlay, cartoon, ugly, static camera, slow motion, still frame, frozen, text, watermark, deformed jet, wings melting, camera pulls back, jet shrinks into distance",
        hold_mount="The camera is bolted inside the pressurized cockpit looking directly at the pilot or over their shoulder, so the pilot, helmet, controls, and canopy frame stay rock solid in frame while canyon walls and clouds blur past at Mach speed.",
        hold_chase="The camera stays locked tight to the cockpit canopy through high-G maneuvers, framing the pilot clearly inside.",
        hold_trackside="The tracking camera snaps violently tracking the cockpit through the flyby.",
        sequence=[
            ("pilot face-cam", "high-g strain"),
            ("over-the-shoulder pilot pov", "canyon trench run"),
            ("crash zoom to cockpit", "mach punch"),
            ("helmet visor reflection", "canyon trench run"),
            ("cockpit 360 dynamic orbit", "combat formation"),
            ("cockpit-to-exterior punch through", "afterburner climb"),
        ],
        auto_order=["pilot face-cam", "over-the-shoulder pilot pov", "crash zoom to cockpit", "helmet visor reflection", "cockpit 360 dynamic orbit", "cockpit-to-exterior punch through"],
    ),
    "action_sports": dict(
        label="Downhill MTB & Freeride",
        sound="dirt spraying, knobby tires tearing through gravel, suspension bottoming out, heavy rider breathing and wind rush",
        default_subject="an original carbon full-suspension downhill mountain bike painted gloss acid yellow and raw carbon with dual-crown front suspension forks, "
                        "fat knobby downhill tires and yellow mudguard, ridden by an aggressive freeride rider in a dark downhill jersey, armored pads and full-face helmet with tinted goggles",
        default_rival="an original pro downhill mountain bike painted matte stealth black with red suspension linkages, carbon rims, "
                      "ridden by a competing freeride rider in a red-and-black racing jersey and full-face helmet",
        no_brand="Original fictional mountain bike designs with no real brand logos, no sponsor emblems, and no readable text.",
        plate_neg="blurry, lowres, watermark, logo, motor, motorcycle, engine, car, extra wheels, deformed rider, missing pedals, cartoon, CGI render, video game HUD, split screen",
        video_neg="pc game, video game, cartoon, ugly, static camera, slow motion, still frame, frozen, text, watermark, deformed bike, rider falls off, camera pulls back, bike shrinks into distance",
        hold_mount="The camera is clamped to the bike frame or handlebars, so the handlebars and front fork stay locked in the foreground the whole time while roots, rocks, and trees whip past in heavy motion blur.",
        hold_chase="The FPV chase drone threads between trees inches behind the rider's helmet, keeping the rider centered and dynamic through every berm and jump.",
        hold_trackside="The low trailside camera stays fixed as dirt and roost blast toward the lens when the rider carves the corner.",
        sequence=[
            ("mtb handlebar chesty", "rock garden blitz"),
            ("mtb fork stanchion", "berm railing"),
            ("fpv drone dive", "big gap jump"),
            ("mtb rear frame", "roost blowout"),
            ("roadside whip pan", "finish drop"),
        ],
        auto_order=["mtb handlebar chesty", "mtb fork stanchion", "fpv drone dive", "mtb rear frame", "roadside whip pan"],
    ),
    "speedboat": dict(
        label="Offshore Powerboats & Catamarans",
        sound="twin high-performance marine outboards screaming at 8000 RPM, hull slapping water, roaring rooster tail wash",
        default_subject="an original twin-hull offshore racing catamaran powerboat painted gloss electric cobalt blue with white racing stripes, "
                        "carbon rear aero wing, massive dual rooster-tail water spray, and bold white racing number 5 on the sponsons, "
                        "piloted by a powerboat racer in closed cockpit with safety harness and helmet",
        default_rival="an original offshore deep-V racing powerboat painted gloss molten orange with chrome exhausts, black deck, "
                      "and bold racing number 8 on the hull",
        no_brand="Original fictional racing powerboat designs with no brand logos, no sponsor emblems, and no readable text.",
        plate_neg="blurry, lowres, watermark, logo, text, sailboat, yacht, cargo ship, wheels, cartoon, CGI render, split screen",
        video_neg="pc game, video game, cartoon, ugly, static camera, slow motion, still frame, frozen, text, watermark, deformed boat, camera pulls back, boat shrinks into distance",
        hold_mount="The camera is bolted to the powerboat sponson or deck, so the boat's hull stays locked in frame while massive sheets of water spray, rooster tails, and the ocean tear past at 120 knots.",
        hold_chase="The chase helicopter or drone flies low inches above the ocean spray, keeping the racing boat large and centered in frame.",
        hold_trackside="The buoy tracking camera tracks the racing powerboat skimming across the water at extreme speed.",
        sequence=[
            ("boat bow skim", "wave skip launch"),
            ("boat cockpit canopy", "head-to-head drag"),
            ("boat transom wake", "rooster tail blast"),
            ("boat sponson side", "buoy turn carve"),
            ("helicopter wide", "open water sprint"),
        ],
        auto_order=["boat bow skim", "boat cockpit canopy", "boat transom wake", "boat sponson side", "helicopter wide"],
    ),
}

# ----------------------------------------------------------------------------------------------------- worlds
WORLDS = {
    "mountain pass": dict(
        scene="a narrow alpine mountain road of tight hairpin switchbacks with stone guard walls and steel guardrails, "
              "a sheer drop to a green valley, pine forest and snow-capped peaks",
        light="crisp early morning sun, long shadows, thin mist in the valley",
        sound="twin engines echoing off the rock walls"),
    "city night": dict(
        scene="an empty downtown street circuit at night between glass towers, glowing neon bokeh, overpasses, "
              "concrete barriers and wet asphalt mirroring the lights",
        light="night, neon magenta and cyan reflections on wet road, sodium street lamps, light haze",
        sound="high-revving engines bouncing between the buildings"),
    "coastal cliffs": dict(
        scene="a winding two-lane road carved into sea cliffs high above a deep blue ocean, white guardrails, "
              "a lighthouse on the headland",
        light="golden hour sun low over the ocean, warm glints and lens flare",
        sound="engines roaring mixed with wind and distant surf"),
    "city tunnel": dict(
        scene="a long curving city road tunnel with tiled walls and rows of ceiling lights, the exit glowing ahead",
        light="rows of orange tunnel lights strobing across the bodywork",
        sound="exhaust roar booming and reverberating in the tunnel"),
    "city day": dict(
        scene="a closed-off downtown boulevard between steel and glass skyscrapers, concrete overpasses, "
              "kerbs and empty crossings",
        light="bright midday sun, hard shadows, reflections sliding across glass facades",
        sound="engines screaming through the street canyon"),
    "desert canyon": dict(
        scene="a long ribbon of empty highway through red rock canyon country, mesas, scrub and heat shimmer",
        light="harsh late-afternoon sun, warm orange rock, deep blue sky",
        sound="engines howling in open air, gravel pinging"),
    "forest gravel": dict(
        scene="a fast gravel rally stage through a dense pine forest, dust hanging between the trees, crests and ruts",
        light="sun shafts cutting through the trees and the dust",
        sound="rally engines snarling, gravel spraying, stones hitting the underbody"),
    "snow mountain": dict(
        scene="a plowed mountain pass through deep snow banks, frosted pines and a frozen lake below",
        light="cold blue winter light, bright snow glare",
        sound="engines roaring muffled by the snow, tires crunching"),
    "rain docklands": dict(
        scene="an industrial dockland circuit between stacked shipping containers and cranes, puddles on concrete",
        light="night, heavy rain, sodium orange lamps, cold blue shadows, spray and reflections",
        sound="engines roaring, rain hammering, water hissing under the tires"),
    "race circuit": dict(
        scene="a professional motorsport circuit with red and white kerbs, run-off areas, tire walls and grandstands",
        light="clear late-afternoon golden sun",
        sound="high-revving race engines, rapid gear shifts, tire squeal"),
    "island road course": dict(
        scene="a narrow coastal mountain road course through stone-walled villages, crests, and mountain roads with the sea glistening below",
        light="crisp morning golden sunlight, tree shadows flickering across the tarmac",
        sound="screaming high-revving 4-cylinder superbike engine echoing off stone cottages and rock faces"),
    "supersonic canyon": dict(
        scene="a vast red-rock desert trench canyon with towering sheer vertical sandstone cliffs, jagged spires and heat shimmer",
        light="intense late-afternoon sun blasting off red sandstone walls against a deep cobalt sky",
        sound="deafening twin turbofan roar and sonic boom shockwave reverberating through the canyon walls"),
    "stratosphere clouds": dict(
        scene="high altitude above an unbroken sea of billowing golden clouds at sunset, deep purple sky fading into black space above",
        light="dramatic sunset alpenglow, gleaming reflections across the fuselage and canopy",
        sound="deep howling jet turbine rush, cockpit oxygen and wind friction"),
    "whistler forest": dict(
        scene="a steep alpine pine forest freeride mountain bike track with massive wooden berms, roots, rock slabs, and loamy dark soil",
        light="dappled sunlight filtering through tall evergreen trees, airborne dust motes",
        sound="knobby tires tearing through loam, suspension hissing, freehub buzzing"),
    "miami offshore": dict(
        scene="open turquoise ocean waters off a sun-drenched tropical coastline, distant city towers and coral reefs glowing below crystal clear water",
        light="brilliant midday tropical sun, glittering water highlights and lens flare",
        sound="twin 1500hp offshore racing V8s roaring at redline, massive rooster tail splashing"),
    "fjord glass": dict(
        scene="a calm deep blue glacial fjord flanked by 3000-foot sheer granite cliffs and cascading waterfalls, glassy mirror water surface",
        light="clear Nordic blue hour light, mist hovering over the water line",
        sound="deep reverberating engine echo across the fjord, water hissing off the hull"),
}

LIGHTS = {
    "world default": None,
    "golden hour": "golden hour, low warm sun, long shadows, lens flare",
    "blue hour": "blue hour dusk, deep blue sky, headlights and tail lights glowing",
    "night": "night, headlights cutting through the dark, glowing tail lights, street lamps",
    "midday": "bright midday sun, hard shadows, high contrast",
    "overcast": "soft overcast light, moody grey sky, rich colors",
    "rain": "pouring rain, wet glossy road, spray off the tires, droplets on the lens",
    "fog": "thick fog, headlight beams, silhouettes fading into haze",
    "sunrise": "sunrise, pink and orange sky, sun cresting the horizon",
}

# ----------------------------------------------------------------------------------------------------- action beats
# Each action has a race version (with {rival}) and a solo version (when rival is disabled).
ACTIONS = {
    "launch": dict(
        plate="launching off the line side-by-side with {rival}, rear tires spinning in thick clouds of white tire smoke",
        motion="Both cars launch from a standstill side-by-side, rear tires spinning and pouring smoke, then hook up and rocket forward neck-and-neck",
        solo_plate="launching from a standstill, rear tires spinning in thick white smoke",
        solo_motion="The car launches from a standstill, rear tires spinning and smoking, then hooks up and rockets away"),
    "drafting": dict(
        plate="tucked inches behind the rear diffuser of {rival} in a high-speed slipstream flat out on a straight",
        motion="The car drafts inches behind {rival} at extreme speed, weaving slightly in the slipstream as both cars scream down the straight",
        solo_plate="flat out on a straight at maximum speed",
        solo_motion="The car is flat out, accelerating hard down the straight"),
    "flat out": dict(
        plate="charging flat out on a straight with {rival} chasing half a car-length behind",
        motion="The car blasts flat out down the straight at redline with {rival} locked in hot pursuit right behind it",
        solo_plate="flat out on a straight",
        solo_motion="The car is flat out, accelerating hard down the road"),
    "carve a bend": dict(
        plate="leaning hard into a fast sweeping bend nose-to-tail with {rival}",
        motion="The car carves through a fast sweeping bend at the limit of grip with {rival} right on its tail",
        solo_plate="leaning into a fast sweeping bend",
        solo_motion="The car carves through a fast sweeping bend, body rolling, tires gripping at the limit"),
    "hard brake": dict(
        plate="braking hard into a corner beside {rival}, carbon-ceramic brake discs glowing bright orange and exhaust popping blue-orange flame",
        motion="Both cars stamp on the brakes into the corner, noses diving, brake discs glowing bright orange and flames popping from the exhaust on the downshift",
        solo_plate="braking hard into a corner, brake discs glowing orange and exhaust popping flame",
        solo_motion="The car stamps on the brakes, the nose dives, brake discs glow orange and flames pop from the exhaust on the downshift"),
    "hairpin": dict(
        plate="diving into the inside apex of a tight hairpin alongside {rival}, inside front wheel kissing the red-and-white kerb",
        motion="The car dives late onto the inside apex of the tight hairpin alongside {rival}, tires scrubbing the kerb, and powers out hard",
        solo_plate="sliding around a tight hairpin, inside wheel brushing the kerb",
        solo_motion="The car brakes late, slides around the tight hairpin and launches out of it with the rear squirming"),
    "drift": dict(
        plate="drifting sideways at opposite lock around the outside of {rival} through a corner with thick white tire smoke pouring off the rear wheels",
        motion="The car flicks sideways into an aggressive full-lock drift around {rival} through the corner, rear tires pouring white smoke, then snaps straight and fires ahead",
        solo_plate="drifting sideways through a corner with tire smoke pouring off the rear wheels",
        solo_motion="The car flicks sideways into a full drift, rear tires pouring white smoke, front wheels at opposite lock, then snaps straight and fires out of the corner"),
    "overtake": dict(
        plate="pulling alongside {rival} door-to-door to overtake at full throttle, the two cars inches apart",
        motion="The car pulls out from behind {rival}, draws door-to-door inches apart, and surges past into the lead at redline",
        solo_plate="accelerating hard out of a corner at full throttle",
        solo_motion="The car surges forward at full throttle, suspension hunkered down"),
    "lead defense": dict(
        plate="holding the lead flat out while {rival} lunges right behind the rear bumper in hot pursuit",
        motion="The car rockets ahead defending the lead while {rival} weaves aggressively right behind the rear bumper in the slipstream",
        solo_plate="accelerating away at full throttle",
        solo_motion="The car rockets away at full speed"),
    "near miss": dict(
        plate="threading a narrow gap between {rival} and the guardrail at full speed, sparks flying",
        motion="The car threads a razor-thin gap between {rival} and the barrier at full speed, sparks showering off the guardrail",
        solo_plate="threading a gap between barriers at full speed",
        solo_motion="The car threads a narrow gap at full speed, missing the barrier by inches, sparks flying"),
    "finish flyby": dict(
        plate="blasting past at maximum speed half a car-length ahead of {rival} in a side-by-side drag race to the line",
        motion="The car screams past half a car-length ahead of {rival} at maximum speed, air shockwave kicking up dust and sparks",
        solo_plate="blasting past at maximum speed",
        solo_motion="The car screams past at maximum speed in a blur"),
    "jump": dict(
        plate="cresting a rise side-by-side with {rival} with the front wheels lifting off the asphalt",
        motion="Both cars crest a rise at speed, wheels going light, land hard with suspension compressing and sparks flying, and keep accelerating",
        solo_plate="cresting a rise with the front wheels going light",
        solo_motion="The car crests a rise, all four wheels leave the road, it lands hard, suspension compressing, and keeps accelerating"),
    # motorcycle actions
    "knee-down carve": dict(
        plate="leaning at an extreme 60-degree angle with the rider knee puck scraping the kerb beside {rival}",
        motion="The superbike tips aggressively into full lean, knee puck throwing sparks off the kerb beside {rival}, and throttles out hard",
        solo_plate="leaning at an extreme 60-degree angle with the knee puck scraping the kerb",
        solo_motion="The superbike tips aggressively into full lean, knee puck scraping the kerb, and throttles out hard"),
    "wheelie launch": dict(
        plate="launching off the line with the front wheel lifting into a controlled power wheelie alongside {rival}",
        motion="The superbike launches hard off the grid, front tire floating inches above the track beside {rival}, then snaps down as it accelerates away",
        solo_plate="launching with the front wheel lifting into a controlled power wheelie",
        solo_motion="The superbike launches hard off the grid, front tire floating in a power wheelie, then snaps down and rockets away"),
    "slipstream pass": dict(
        plate="tucked tight in the aerodynamic slipstream inches behind {rival} at 200 mph",
        motion="Tucked low behind the bubble screen in the slipstream of {rival}, pulls out and slices past on the brakes into the turn",
        solo_plate="tucked flat out in maximum aerodynamic tuck at 200 mph",
        solo_motion="Rider in full aero tuck screaming down the straight at maximum speed"),
    "hard trail brake": dict(
        plate="braking late into the apex alongside {rival}, front fork fully compressed and rear wheel lifting slightly",
        motion="Trail braking deep into the corner beside {rival}, front suspension fully loaded, rear tire squirming on the limit",
        solo_plate="braking late into the apex with the front fork fully compressed",
        solo_motion="Trail braking deep into the turn, front suspension fully loaded, tires scrubbing at the limit"),
    "high-speed tuck": dict(
        plate="tucked into a full aerodynamic tuck flat out on the straight neck-and-neck with {rival}",
        motion="Rider in a full tuck behind the windscreen screaming down the straight neck-and-neck with {rival} at redline",
        solo_plate="tucked in a full aerodynamic tuck flat out on the straight",
        solo_motion="Rider in full tuck behind the windscreen accelerating down the straight at redline"),
    "quickshifter pop": dict(
        plate="accelerating through the gears beside {rival}, titanium exhaust spitting blue flames on upshifts",
        motion="Rapid-fire clutchless upshifts with loud exhaust flame pops, rocketing past {rival} at full acceleration",
        solo_plate="accelerating through the gears with the titanium exhaust spitting blue flame",
        solo_motion="Rapid-fire upshifts with loud exhaust flame pops, accelerating hard down the track"),
    # jet actions
        "high-g strain": dict(
        plate="pulling an extreme 9-G banking turn alongside {rival}, the pilot tensing against intense gravity forces, G-suit inflated, vapor trails streaming across the canopy",
        motion="The aircraft snaps into an intense 9-G bank, the pilot fighting extreme gravity, head straining, eyes locked forward, cockpit rattling furiously as the aircraft slices through turbulence at Mach 1.5",
        solo_plate="pulling an extreme 9-G banking turn, the pilot tensing against intense gravity forces, G-suit inflated, vapor trails streaming",
        solo_motion="The aircraft snaps into an intense 9-G bank, the pilot fighting extreme gravity, head straining, eyes locked forward, cockpit rattling furiously"),
    "canyon trench run": dict(
        plate="carving at Mach 1 through a narrow trench canyon alongside {rival}, wingtips inches from the vertical sandstone walls",
        motion="Twin supersonic fighter jets slice through the narrow canyon trench, banking knife-edge around jagged spires with shockwaves trailing",
        solo_plate="carving at Mach 1 through a narrow trench canyon, wingtips inches from the rock walls",
        solo_motion="The fighter jet slices through the narrow canyon trench, banking knife-edge around rock spires with shockwaves trailing"),
    "afterburner climb": dict(
        plate="pulling into a vertical 9G climb into the sky with glowing afterburners side-by-side with {rival}",
        motion="Both jets pull straight up into a vertical 9G rocket climb, twin afterburners glowing bright cyan and orange, punching through the clouds",
        solo_plate="pulling into a vertical 9G climb with glowing afterburners punching through the clouds",
        solo_motion="The jet pulls straight up into a vertical 9G rocket climb, afterburners glowing, punching vertically through the cloud deck"),
    "mach punch": dict(
        plate="punching through the sound barrier with a conical Prandtl-Glauert vapor cone forming around the fuselage beside {rival}",
        motion="The jet breaks Mach 1, a white conical vapor shockwave flashes around the wings, and it accelerates away in a sonic blast",
        solo_plate="punching through the sound barrier with a conical vapor cone around the fuselage",
        solo_motion="The jet breaks Mach 1, a white conical vapor shockwave flashes around the wings, and it rockets away in a sonic blast"),
    "barrel roll evade": dict(
        plate="executing a high-speed barrel roll around {rival} through the clouds, contrails spiraling off the wingtips",
        motion="The fighter jet rolls 360 degrees smoothly over {rival}, wingtip contrails corkscrewing through the sky as it snaps back onto target",
        solo_plate="executing a high-speed barrel roll through the clouds, contrails spiraling off the wingtips",
        solo_motion="The fighter jet rolls 360 degrees smoothly, wingtip contrails corkscrewing through the sky"),
    "combat formation": dict(
        plate="flying in tight supersonic tactical combat formation meters off the wing of {rival}",
        motion="Both fighter jets hold razor-sharp supersonic formation through atmospheric turbulence, sunlight glinting off the canopies",
        solo_plate="flying at supersonic speed in combat patrol trim",
        solo_motion="The jet cruises at supersonic speed, canopy reflections shifting smoothly"),
    "supersonic flyby": dict(
        plate="screaming past at Mach 1.5 meters above the terrain, kicking up a massive supersonic shockwave",
        motion="Low-altitude supersonic pass, sonic boom shockwave blasting across the terrain as the jet disappears toward the horizon",
        solo_plate="screaming past at Mach 1.5 meters above the ground",
        solo_motion="Low-altitude supersonic pass, sonic boom shockwave blasting across the ground"),
    # action sports / downhill mtb actions
    "rock garden blitz": dict(
        plate="smashing through a steep technical rock garden at high speed beside {rival}, suspension forks fully compressed",
        motion="The downhill bike charges through boulders and tree roots at speed, suspension working violently beside {rival}",
        solo_plate="smashing through a steep technical rock garden at high speed",
        solo_motion="The downhill bike charges through boulders and tree roots at speed, suspension working violently over the rocks"),
    "berm railing": dict(
        plate="railing a high-speed dirt berm at full lean with dirt roost exploding off the rear tire beside {rival}",
        motion="The bike carves deep into the banked dirt berm, throwing a huge wall of roost and blasting out of the corner beside {rival}",
        solo_plate="railing a high-speed dirt berm at full lean with roost exploding off the rear tire",
        solo_motion="The bike carves deep into the banked dirt berm, throwing a huge wall of roost and blasting out of the turn"),
    "big gap jump": dict(
        plate="soaring through the air on a massive 40-foot canyon gap jump side-by-side with {rival}, pine trees below",
        motion="Both riders soar across the massive jump, wheels spinning in mid-air, land smoothly and pump into the next roller",
        solo_plate="soaring through the air on a massive 40-foot canyon gap jump",
        solo_motion="The rider soars across the massive jump, wheels spinning in mid-air, lands smoothly on the transition"),
    "roost blowout": dict(
        plate="drifting the rear tire through loose dirt and gravel, throwing a roost fan toward the camera beside {rival}",
        motion="The rider locks the rear brake, flicks the bike sideways into a roost spray, and pedals hard out of the turn",
        solo_plate="drifting the rear tire through loose dirt, throwing a roost fan toward the camera",
        solo_motion="The rider flicks the bike sideways into a roost spray and pedals hard out of the turn"),
    "finish drop": dict(
        plate="dropping off a 15-foot wooden cliff drop into the finish arena beside {rival}",
        motion="Plunging off the wooden cliff drop, landing hard on the transition and rolling across the finish line",
        solo_plate="dropping off a 15-foot wooden cliff drop into the finish arena",
        solo_motion="Plunging off the wooden cliff drop, landing hard on the transition and rolling across the finish line"),
    # speedboat actions
    "wave skip launch": dict(
        plate="skimming across the ocean chop at 120 knots with both sponsons lifting off the water beside {rival}",
        motion="The offshore catamaran skips across the waves at breakneck speed, hulls airing out and landing cleanly beside {rival}",
        solo_plate="skimming across the ocean chop at 120 knots with both sponsons lifting off the water",
        solo_motion="The offshore catamaran skips across the waves at breakneck speed, hulls airing out and landing cleanly"),
    "rooster tail blast": dict(
        plate="blasting across the open water throwing a 50-foot rooster tail of white spray behind {rival}",
        motion="Both powerboats scream down the straightaway, massive rooster tails arching into the sky, twin engines howling",
        solo_plate="blasting across the open water throwing a 50-foot rooster tail of white spray",
        solo_motion="The powerboat screams down the straightaway, massive rooster tail arching into the sky, engines howling"),
    "buoy turn carve": dict(
        plate="carving a tight 90-degree turn around the yellow race buoy with water spray sheeting off the hull inches from {rival}",
        motion="Hard turn around the turn buoy, sponson digging into the ocean, wall of water spray blasting outward",
        solo_plate="carving a tight 90-degree turn around the race buoy with water spray sheeting off the hull",
        solo_motion="Hard turn around the buoy, sponson digging in, wall of water spray blasting outward"),
    "head-to-head drag": dict(
        plate="locked side-by-side at 130 mph on glassy water, hulls skipping and air cushions packing beneath {rival}",
        motion="Dead heat drag race across the water surface, spray and sunlight flashing as both powerboats fight for the lead",
        solo_plate="full throttle drag sprint at 130 mph across glassy water",
        solo_motion="Full throttle drag sprint across glassy water, spray and sunlight flashing"),
    "open water sprint": dict(
        plate="full throttle open water sprint towards the horizon, twin hulls dancing over the surface alongside {rival}",
        motion="Full throttle sprint across the open water, wake unspooling in a straight line behind as the boat accelerates",
        solo_plate="full throttle open water sprint towards the horizon",
        solo_motion="Full throttle sprint across the open water, wake unspooling in a straight line behind"),
}

# ----------------------------------------------------------------------------------------------------- camera rigs
RIGS = {
    "wheel cam (front)": dict(
        group="mount", lens="ultra-wide action camera",
        plate="extreme low close-up from a camera clamped inside the front wheel arch of {car}: the spinning front tire "
              "and glowing orange brake caliper fill the left side of the frame, the asphalt just centimeters below the lens, "
              "the road of {world} stretching ahead, {action}",
        motion="Camera rigidly mounted on the front wheel arch, locked to the car. The front tire spins in a fierce blur, "
               "the asphalt streams past centimeters below the lens with heavy motion blur, kerbs and road lines whip by, "
               "the suspension shudders and the frame vibrates with every bump.",
        actions=["hairpin", "hard brake", "carve a bend", "overtake"],
        fx=dict(shake=0.7, speed_blur=0.65, zoom_punch=0.0, chroma=0.3), seconds=3),
    "wheel cam (rear)": dict(
        group="mount", lens="ultra-wide action camera",
        plate="low close-up from a camera clamped behind the rear wheel of {car}, looking forward along the sculpted flank: "
              "the fat rear tire in the lower foreground, the side of the car leading forward, {world} beyond, {action}",
        motion="Camera mounted low behind the rear wheel, locked to the car. The rear tire spins and throws grit and "
               "white smoke toward the lens, the road surface tears past in a blur, the body hunkers down as the car accelerates, "
               "the frame shakes hard.",
        actions=["launch", "drift", "overtake", "flat out"],
        fx=dict(shake=0.8, speed_blur=0.65, zoom_punch=0.15, chroma=0.3), seconds=3),
    "door mount": dict(
        group="mount", lens="wide",
        plate="camera suction-mounted on the rear quarter panel of {car} looking forward along the side of the car past the "
              "side mirror and front wheel toward the road ahead, {world}, {action}",
        motion="Side-mounted camera locked to the bodywork. The flank of the car stays rock steady on one side of the frame while "
               "the road and rival smear past with intense speed, guardrails streaking into lines, reflections racing across the "
               "paint, the mount buzzing with engine vibration.",
        actions=["overtake", "near miss", "carve a bend", "flat out"],
        fx=dict(shake=0.45, speed_blur=0.55, zoom_punch=0.1, chroma=0.2), seconds=3),
    "bumper cam": dict(
        group="mount", lens="wide",
        plate="very low camera mounted on the front bumper splitter of {car}, a few centimeters above the asphalt, looking "
              "straight ahead down {world}, the carbon tip of the front splitter visible along the bottom edge, {action}",
        motion="Bumper camera skimming centimeters above the road at extreme speed. Lane markings and asphalt seams whip under "
               "the lens, the rival car and corners rush straight at the camera, intense high-speed vibration.",
        actions=["drafting", "overtake", "near miss", "hairpin"],
        fx=dict(shake=0.8, speed_blur=0.75, zoom_punch=0.3, chroma=0.3), seconds=3),
    "rear bumper (reverse)": dict(
        group="mount", lens="wide",
        plate="reverse-facing camera mounted on the rear bumper of {car} looking backward: the glowing tail lights and quad "
              "exhaust tips at the bottom edge of the frame, the road of {world} unspooling behind, {action}",
        motion="Reverse-facing camera on the rear bumper. The road pours out from under the car at tremendous speed while "
               "the chasing rival surges in the slipstream behind, exhaust tips spitting flame, heat haze rippling.",
        actions=["lead defense", "hard brake", "drift", "flat out"],
        fx=dict(shake=0.65, speed_blur=0.6, zoom_punch=0.0, chroma=0.25), seconds=2),
    "roof cam": dict(
        group="mount", lens="wide",
        plate="camera mounted low on the roof of {car} looking forward over the windshield and long hood, the road of "
              "{world} curving ahead, {action}",
        motion="Roof-mounted camera riding with the car. The hood points aggressively into each corner, the horizon tilts as "
               "the car rolls through the bends, the road surges toward the lens, wind buffets the mount.",
        actions=["carve a bend", "overtake", "hairpin", "jump"],
        fx=dict(shake=0.5, speed_blur=0.5, zoom_punch=0.0, chroma=0.15), seconds=3),
    "exhaust close-up": dict(
        group="mount", lens="macro",
        plate="tight low close-up of the quad exhaust tips, rear diffuser and rear tire of {car}, glowing hot with blue-orange "
              "exhaust flame, {world} motion-blurred behind, {action}",
        motion="Tight camera locked on the exhaust tips and rear diffuser as the car downshifts: bright flames burst from the pipes, "
               "heat shimmer warps the air, the diffuser vibrates violently, the asphalt blurs past below.",
        actions=["hard brake", "launch", "lead defense"],
        fx=dict(shake=0.6, speed_blur=0.4, zoom_punch=0.45, chroma=0.3), seconds=2),
    "cockpit pov": dict(
        group="mount", lens="ultra-wide action camera",
        plate="interior roll-cage POV from inside the cockpit of {car} looking out through the windshield: the carbon racing "
              "steering wheel with glowing red shift-light LEDs and dashboard in the lower foreground, the hood and {world} "
              "ahead through the glass, {action}",
        motion="Cockpit roll-cage camera bolted inside the cabin. The dashboard and roll-cage stay locked in frame while the "
               "steering wheel makes rapid corrections, red shift lights flash at redline, and the road and rival car rush "
               "past outside the windshield with intense vibration.",
        actions=["drafting", "overtake", "hairpin", "near miss"],
        fx=dict(shake=0.65, speed_blur=0.55, zoom_punch=0.15, chroma=0.2), seconds=3),
    "side mirror cam": dict(
        group="mount", lens="wide",
        plate="camera suction-mounted on the door just behind the carbon-fiber side mirror of {car}, framing both the road "
              "ahead along the front fender and the side mirror glass reflecting the chasing rival behind in {world}, {action}",
        motion="Door-mounted camera locked right behind the side mirror. The mirror housing and front fender stay fixed while "
               "asphalt and barriers streak past at extreme speed and the rival looms in the mirror reflection.",
        actions=["lead defense", "overtake", "drift", "hard brake"],
        fx=dict(shake=0.5, speed_blur=0.55, zoom_punch=0.1, chroma=0.2), seconds=3),
    "underbody cam": dict(
        group="mount", lens="ultra-wide action camera",
        plate="extreme ground-hugging camera mounted under the front chassis of {car} between the spinning front tires just "
              "millimeters above the asphalt, titanium skid-block sparks showering off the road surface of {world}, {action}",
        motion="Underbody camera millimeters above the rushing tarmac between the front wheels. The road surface tears directly "
               "underneath the lens in a violent blur, suspension arms work hard, and bright sparks shower toward the camera.",
        actions=["launch", "flat out", "jump", "hard brake"],
        fx=dict(shake=0.85, speed_blur=0.8, zoom_punch=0.2, chroma=0.35), seconds=2),
    "hood cam": dict(
        group="mount", lens="wide",
        plate="camera bolted to the center of the hood of {car} near the windshield base, looking forward over the sculpted "
              "hood vents and front fenders into {world}, {action}",
        motion="Hood-mounted camera locked to the centerline of the car. The sculpted hood fills the lower frame, heat shimmer "
               "rising from the vents as the nose dives and pitches through the corners at breakneck speed.",
        actions=["carve a bend", "overtake", "hairpin", "drafting"],
        fx=dict(shake=0.55, speed_blur=0.6, zoom_punch=0.1, chroma=0.2), seconds=3),
    "arm car chase": dict(
        group="chase", lens="wide",
        plate="low tracking shot from a Russian-arm crane on a chase car, just behind and to the rear quarter of {car} at bumper "
              "height in {world}, {action}",
        motion="Camera on a Russian-arm crane mounted to a chase car, holding low off the rear quarter at bumper height and "
               "matching the cars' extreme speed. The arm swings smoothly closer, the background streaks into horizontal blur, "
               "the wheels spin fast, the hero car stays sharp.",
        actions=["drafting", "overtake", "drift", "carve a bend"],
        fx=dict(shake=0.35, speed_blur=0.55, zoom_punch=0.0, chroma=0.15), seconds=3),
    "arm car orbit": dict(
        group="chase", lens="wide",
        plate="dramatic low three-quarter view of {car} from a crane arm on a chase car, close and low in {world}, {action}",
        motion="The crane arm sweeps fast around the moving car from the rear three-quarter along the flank while racing at "
               "high speed. The background spins and blurs, light slides over the bodywork, the car stays locked in the center.",
        actions=["overtake", "flat out", "carve a bend"],
        fx=dict(shake=0.25, speed_blur=0.55, zoom_punch=0.0, chroma=0.15), seconds=3),
    "fpv drone dive": dict(
        group="chase", lens="fisheye-wide",
        plate="steep low-altitude FPV racing drone view banking right behind {car} through {world}, {action}",
        motion="FPV racing drone dives steeply toward the asphalt and chases tight behind the cars, banking hard through the "
               "corner with them, the horizon tilting dynamically, tire smoke and terrain rushing past the lens.",
        actions=["drift", "hairpin", "carve a bend", "overtake"],
        fx=dict(shake=0.45, speed_blur=0.6, zoom_punch=0.25, chroma=0.2), seconds=3),
    "fpv drone chase": dict(
        group="chase", lens="fisheye-wide",
        plate="FPV drone shot a few meters behind and slightly above {car} as it tears through {world}, {action}",
        motion="FPV drone locked inches behind the rear wing, banking and weaving with the car at breakneck speed, "
               "tire smoke and sparks streaming straight past the camera.",
        actions=["drift", "overtake", "near miss", "flat out"],
        fx=dict(shake=0.45, speed_blur=0.6, zoom_punch=0.2, chroma=0.2), seconds=3),
    "top-down drone": dict(
        group="chase", lens="long",
        plate="bird's-eye view looking straight down from directly above, a 90-degree overhead drone shot of {car} in {world}, "
              "{action}",
        motion="Overhead drone locked straight down, tracking the cars as they slice through the corner, road markings, "
               "tire smoke and shadows streaming beneath, the drone rotating smoothly as the cars turn.",
        actions=["drift", "hairpin", "overtake"],
        fx=dict(shake=0.15, speed_blur=0.35, zoom_punch=0.0, chroma=0.05), seconds=3),
    "helicopter wide": dict(
        group="chase", lens="long telephoto",
        plate="sweeping aerial helicopter view of {world}, {car} racing along the road, the epic landscape filling the "
              "frame, {action}",
        motion="Aerial helicopter shot banking alongside the road, the cars tearing through the landscape, the camera "
               "gliding fast and descending toward the action.",
        actions=["flat out", "carve a bend", "hairpin"],
        fx=dict(shake=0.15, speed_blur=0.25, zoom_punch=0.0, chroma=0.0), seconds=3),
    "roadside whip pan": dict(
        group="trackside", lens="long",
        plate="low roadside camera at the edge of the kerb in {world}, {car} charging straight toward the lens at full speed, "
              "{action}",
        motion="Low roadside camera at the kerb. The cars charge toward the lens, roar past inches away, and the camera "
               "whip-pans violently to follow them as they rocket away, the background smearing with heavy horizontal motion blur.",
        actions=["finish flyby", "overtake", "near miss", "flat out"],
        fx=dict(shake=0.65, speed_blur=0.75, zoom_punch=0.15, chroma=0.3), seconds=2),
    "low pass-by": dict(
        group="trackside", lens="ultra-wide",
        plate="ground-level camera at the edge of the asphalt in {world}, the red-and-white kerb huge in the foreground, "
              "{car} about to blast past, {action}",
        motion="Ground-level camera fixed at the edge of the road. The cars blast past the lens with a violent shockwave of air "
               "and engine roar, filling the frame, grit and sparks swirling in their wake.",
        actions=["finish flyby", "flat out", "drift", "overtake"],
        fx=dict(ramp=1.0, shake=0.75, speed_blur=0.7, zoom_punch=0.0, chroma=0.3), seconds=2),
    "crash zoom": dict(
        group="trackside", lens="telephoto zoom",
        plate="low telephoto shot looking down the straight of {world} as {car} charges head-on toward the camera through heat haze, "
              "{action}",
        motion="Telephoto camera snaps into a violent crash zoom on the lead car's headlights and splitter as it hurtles "
               "toward the lens, heat haze shimmering, the frame vibrating with speed.",
        actions=["hard brake", "overtake", "flat out", "hairpin"],
        fx=dict(shake=0.55, speed_blur=0.5, zoom_punch=1.0, chroma=0.4), seconds=2),
    # motorcycle rigs
    "moto handlebar pov": dict(
        group="mount", lens="ultra-wide action camera",
        plate="action camera mounted on the triple clamp / handlebar of {car} looking forward over the glowing digital TFT race dash "
              "and tiny curved windscreen, rider gloved hands gripping the clip-on bars, {world} rushing ahead, {action}",
        motion="Handlebar-mounted camera locked to the motorcycle triple clamp. The digital dashboard and clip-on bars stay anchored in the lower frame, "
               "the horizon tilts dramatically as the rider countersteers and banks into corners, asphalt smearing past beneath with violent speed.",
        actions=["high-speed tuck", "slipstream pass", "hard trail brake", "quickshifter pop"],
        fx=dict(shake=0.7, speed_blur=0.7, zoom_punch=0.1, chroma=0.25), seconds=3),
    "moto knee puck cam": dict(
        group="mount", lens="ultra-wide action camera",
        plate="extreme low camera mounted on the side aerodynamic fairing of {car} right behind the rider's knee slider puck, "
              "centimeters above the asphalt as the bike leans at an extreme 60-degree angle into {world}, {action}",
        motion="Knee-slider camera millimeters off the track surface. The fairing and scuffed knee puck stay locked in frame while "
               "the asphalt and red-and-white kerbs tear past at 160 mph, friction sparks flying, the bike leaning until the frame almost grazes the ground.",
        actions=["knee-down carve", "hard trail brake", "slipstream pass"],
        fx=dict(shake=0.8, speed_blur=0.8, zoom_punch=0.2, chroma=0.3), seconds=2),
    "moto front fork cam": dict(
        group="mount", lens="wide",
        plate="camera clamped to the lower front suspension fork of {car} looking down at the spinning front wheel, glowing carbon-ceramic brake disc "
              "and tire contact patch, road surface centimeters below in {world}, {action}",
        motion="Front fork camera skimming centimeters above the asphalt. The spinning front wheel and glowing brake rotor fill the left side of frame, "
               "fork stanchions compressing under brutal braking forces, kerbs and road seams whipping under the lens.",
        actions=["hard trail brake", "knee-down carve", "high-speed tuck"],
        fx=dict(shake=0.75, speed_blur=0.7, zoom_punch=0.0, chroma=0.25), seconds=3),
    "moto rear swingarm": dict(
        group="mount", lens="ultra-wide action camera",
        plate="low camera mounted on the aluminum swingarm of {car} next to the spinning fat rear slick tire and titanium exhaust canister, "
              "tire smoke and shredded rubber flying in {world}, {action}",
        motion="Swingarm-mounted camera locked beside the rear tire. The fat racing slick spins in a fierce blur, throwing grit and smoke toward the camera, "
               "exhaust flames popping on downshifts as the bike launches and accelerates hard.",
        actions=["wheelie launch", "quickshifter pop", "lead defense"],
        fx=dict(shake=0.85, speed_blur=0.75, zoom_punch=0.2, chroma=0.35), seconds=2),
    "moto tail reverse": dict(
        group="mount", lens="wide",
        plate="reverse-facing camera on the aerodynamic tail cowl of {car} looking backward: glowing LED tail light in the foreground, "
              "the chasing rival superbike tucked in the slipstream in {world}, {action}",
        motion="Reverse tail-cowl camera. The asphalt pours out behind at dizzying speed while the chasing rival lunges in the slipstream right behind, "
               "the horizon tilting as both bikes sweep through high-speed chicane transitions.",
        actions=["lead defense", "quickshifter pop", "slipstream pass"],
        fx=dict(shake=0.6, speed_blur=0.65, zoom_punch=0.0, chroma=0.2), seconds=3),
    "moto exhaust macro": dict(
        group="mount", lens="macro",
        plate="tight close-up macro of the slash-cut titanium exhaust canister and rear carbon wheel of {car}, blue-orange flame popping "
              "from the exhaust pipe in {world}, {action}",
        motion="Tight macro camera locked to the exhaust pipe as the superbike accelerates: intense blue flames pulse from the canister, "
               "heat shimmer warps the air, the frame vibrating with high-frequency engine buzz.",
        actions=["quickshifter pop", "wheelie launch", "lead defense"],
        fx=dict(shake=0.65, speed_blur=0.5, zoom_punch=0.4, chroma=0.3), seconds=2),
    # fighter jet rigs
        "pilot face-cam": dict(
        group="mount", lens="35mm cinematic close-up",
        plate="cinematic glare-shield camera mounted on top of the instrument panel glare-shield inside {car}, pointing directly back into the face and helmet of the pilot, framing the pilot from the chest up: the pilot focused eyes, flight helmet with visor slightly raised showing their face, oxygen mask, high-G tactical flight suit with patch, through the bubble canopy behind them where {world} rushes past upside-down at supersonic speed, {action}",
        motion="Glare-shield camera locked onto the pilot face inside the pressurized cockpit. The pilot helmet, eyes, and flight suit stay centered in frame, head bracing and tensing against 9G forces as the jet rolls and pulls high-G maneuvers, sun glinting across the canopy glass and HUD reflections flickering across their face while terrain and clouds tear past outside at Mach 2.",
        actions=["high-g strain", "canyon trench run", "barrel roll evade", "mach punch"],
        fx=dict(shake=0.75, speed_blur=0.5, zoom_punch=0.25, chroma=0.25), seconds=3),
    "over-the-shoulder pilot pov": dict(
        group="mount", lens="wide cinematic",
        plate="over-the-shoulder camera mounted just behind the pilot right shoulder inside the cockpit of {car}, framing the pilot flight helmet and shoulder in the foreground, gloved hands gripping the HOTAS flight control stick and twin throttles, glowing digital MFD multi-function displays and HUD flight symbology glowing green on the glass, looking forward through the bubble canopy at {world}, {action}",
        motion="Over-the-shoulder camera locked behind the pilot. The pilot helmet and gloved hands stay anchored in frame, micro-correcting the flight stick as the aircraft slices through turbulence at supersonic speed, green HUD targeting symbology tracking the target ahead while canyon walls blur past outside.",
        actions=["canyon trench run", "combat formation", "mach punch", "afterburner climb"],
        fx=dict(shake=0.6, speed_blur=0.6, zoom_punch=0.15, chroma=0.2), seconds=3),
    "helmet visor reflection": dict(
        group="mount", lens="macro",
        plate="extreme macro close-up of the pilot mirrored gold-tinted helmet visor inside {car}, reflecting {world} and afterburner flames in the curved optical shield with crystal clarity, the oxygen mask and helmet shell framing the shot, {action}",
        motion="Extreme macro on the reflective helmet visor. The pilot breathing moves the mask slightly, the curved visor mirrors the canyon trench and afterburner fire rushing past at supersonic speed, violent high-frequency cockpit vibration.",
        actions=["canyon trench run", "mach punch", "barrel roll evade"],
        fx=dict(shake=0.7, speed_blur=0.45, zoom_punch=0.35, chroma=0.3), seconds=2),
    "crash zoom to cockpit": dict(
        group="chase", lens="telephoto crash zoom",
        plate="dynamic high-speed tracking shot snapping from an exterior supersonic wide shot of {car} instantly crash-zooming straight into the cockpit canopy glass, framing the pilot focused eyes through the windscreen in {world}, {action}",
        motion="Dynamic Hollywood crash zoom! The camera snaps violently from a wide exterior formation shot, punching at extreme speed straight through the bubble canopy glass to lock onto the pilot helmet and intense focused eyes in the cockpit as the jet roars past at Mach 1.5.",
        actions=["mach punch", "canyon trench run", "combat formation"],
        fx=dict(shake=0.85, speed_blur=0.8, zoom_punch=0.85, chroma=0.4), seconds=2),
    "cockpit 360 dynamic orbit": dict(
        group="chase", lens="wide cinematic",
        plate="dynamic orbital camera circling 360 degrees around the bubble canopy of {car} at supersonic speed in {world}, showing both the pilot inside the cockpit and the exterior wings and afterburner flame, {action}",
        motion="High-speed 360 orbital camera sweeps rapidly around the canopy of the jet while maintaining Mach 1.5 velocity. The pilot is clearly visible inside working the flight controls, while the wings, afterburner cones, and canyon walls rotate dynamically around the frame.",
        actions=["combat formation", "canyon trench run", "barrel roll evade"],
        fx=dict(shake=0.4, speed_blur=0.65, zoom_punch=0.1, chroma=0.2), seconds=3),
    "cockpit-to-exterior punch through": dict(
        group="mount", lens="wide",
        plate="camera starts inches from the pilot eyes inside the cockpit of {car}, pushing forward through the glowing HUD glass and bubble canopy into the open air above {world}, {action}",
        motion="Push-through camera starts locked on the pilot intense gaze inside the cockpit, then accelerates forward right through the HUD and canopy glass into the supersonic exterior slipstream as the jet unleashes full afterburner.",
        actions=["canyon trench run", "afterburner climb", "mach punch"],
        fx=dict(shake=0.7, speed_blur=0.75, zoom_punch=0.5, chroma=0.35), seconds=3),
    "jet canopy pov": dict(
        group="mount", lens="ultra-wide",
        plate="cockpit camera bolted inside the fighter jet canopy of {car} looking forward past the glowing green Heads-Up Display (HUD) "
              "and pilot's flight gloves on the control stick, the pointed radome nose diving through {world}, {action}",
        motion="Canopy camera locked inside the pressurized cockpit. The HUD framing and cockpit canopy arch stay fixed in frame, "
               "g-meter climbing as the jet rolls and pulls high-G maneuvers through clouds and terrain at supersonic speed.",
        actions=["canyon trench run", "afterburner climb", "mach punch", "combat formation"],
        fx=dict(shake=0.6, speed_blur=0.65, zoom_punch=0.15, chroma=0.2), seconds=3),
    "jet wingtip mount": dict(
        group="mount", lens="ultra-wide action camera",
        plate="aerodynamic pod camera bolted to the outer wingtip missile rail of {car} looking inward along the swept wing toward the cockpit fuselage "
              "and twin vertical tails, {world} banking below, {action}",
        motion="Wingtip-mounted camera locked to the wing rail. The razor-sharp swept wing leads across the lower third of the frame, "
               "vapor condensation sheeting across the wing surface as the jet pulls high-G turns and terrain spins dizzyingly beneath.",
        actions=["afterburner climb", "canyon trench run", "barrel roll evade"],
        fx=dict(shake=0.55, speed_blur=0.7, zoom_punch=0.1, chroma=0.25), seconds=3),
    "jet afterburner macro": dict(
        group="mount", lens="macro",
        plate="tight camera mounted on the tail boom of {car} looking directly into the glowing supersonic afterburner nozzle with bright Mach diamonds "
              "and radiant thrust plumes, clouds motion-blurred in {world}, {action}",
        motion="Tail-mounted camera staring directly into the afterburner exhaust: intense glowing cyan and orange shock diamonds expand inside the exhaust plume, "
               "heat ripple distortion tearing through the frame as the jet punches through the sound barrier.",
        actions=["mach punch", "afterburner climb", "supersonic flyby"],
        fx=dict(shake=0.75, speed_blur=0.6, zoom_punch=0.35, chroma=0.35), seconds=2),
    "jet tail forward": dict(
        group="mount", lens="wide",
        plate="camera mounted high on the vertical stabilizer tip of {car} looking forward over the dorsal spine, canopy and delta wings toward {world}, {action}",
        motion="Vertical tail camera looking down the length of the aircraft. The entire stealth airframe stays rock solid in the lower center, "
               "leading-edge vortex contrails spiraling over the wings as the aircraft dives and rolls at Mach speed.",
        actions=["barrel roll evade", "canyon trench run", "combat formation"],
        fx=dict(shake=0.5, speed_blur=0.6, zoom_punch=0.0, chroma=0.15), seconds=3),
    "formation chase": dict(
        group="chase", lens="telephoto",
        plate="tight telephoto formation shot from a wingman chase aircraft flying meters off the starboard wing of {car} in {world}, {action}",
        motion="Chase aircraft holding tight tactical formation. The hero jet stays massive and razor-sharp in frame, wingtips almost touching, "
               "sunlight glittering off the pilot's helmet visor as both aircraft slice through atmospheric turbulence in unison.",
        actions=["combat formation", "barrel roll evade", "afterburner climb"],
        fx=dict(shake=0.3, speed_blur=0.5, zoom_punch=0.0, chroma=0.1), seconds=3),
    "jet belly pass": dict(
        group="mount", lens="ultra-wide",
        plate="extreme low aerodynamic camera mounted under the forward fuselage of {car} looking forward between the air intakes, terrain streaming beneath at Mach 1.5 in {world}, {action}",
        motion="Belly camera skimming meters above the terrain at supersonic speed. Mountains and canyon walls rush directly underneath in a blinding streak, "
               "shockwave dust kicking up behind as the jet screams past.",
        actions=["supersonic flyby", "canyon trench run", "mach punch"],
        fx=dict(shake=0.8, speed_blur=0.85, zoom_punch=0.2, chroma=0.3), seconds=2),
    # action sports / downhill mtb rigs
    "mtb handlebar chesty": dict(
        group="mount", lens="ultra-wide action camera",
        plate="ultra-wide action camera chest-mounted on the rider of {car} looking over the wide carbon handlebars, front suspension fork "
              "and knobby front tire tearing through {world}, {action}",
        motion="Chest-mounted action cam. The carbon handlebars and rider's gloved hands remain centered in frame, working rapidly over rough terrain, "
               "the trail, rocks, and trees whipping past in violent motion blur, landing hard off drops with suspension compressing.",
        actions=["rock garden blitz", "berm railing", "big gap jump", "finish drop"],
        fx=dict(shake=0.8, speed_blur=0.7, zoom_punch=0.15, chroma=0.25), seconds=3),
    "mtb fork stanchion": dict(
        group="mount", lens="wide",
        plate="low camera clamped to the downhill suspension fork stanchion of {car} inches above the spinning knobby front tire, "
              "rocks and trail dirt showering in {world}, {action}",
        motion="Fork-stanchion camera inches above the front wheel. The knobby tire spins furiously, kicking up loam and dust, "
               "suspension fork compressing violently over roots and rocks, dirt berms rushing past the lens.",
        actions=["berm railing", "rock garden blitz", "roost blowout"],
        fx=dict(shake=0.85, speed_blur=0.75, zoom_punch=0.1, chroma=0.3), seconds=2),
    "mtb rear frame": dict(
        group="mount", lens="wide",
        plate="camera clamped to the rear chainstay of {car} near the disc brake rotor and knobby rear tire, dirt roost spraying outward in {world}, {action}",
        motion="Chainstay camera locked beside the rear tire. The rear tire locks and drifts sideways through loose soil, throwing a thick fan of roost toward the lens, "
               "rear shock absorbing heavy landing impacts.",
        actions=["roost blowout", "berm railing", "big gap jump"],
        fx=dict(shake=0.75, speed_blur=0.65, zoom_punch=0.2, chroma=0.25), seconds=2),
    # speedboat rigs
    "boat bow skim": dict(
        group="mount", lens="ultra-wide action camera",
        plate="low waterproof camera mounted on the pointed nose bow of {car} skimming just centimeters above the churning turquoise water in {world}, {action}",
        motion="Bow camera skimming inches above the water at 120 knots. Ocean waves rush directly underneath in a blinding foam spray, "
               "the twin hulls skipping across chop, water droplets streaking across the lens, intense high-speed water shudder.",
        actions=["wave skip launch", "head-to-head drag", "open water sprint"],
        fx=dict(shake=0.8, speed_blur=0.8, zoom_punch=0.15, chroma=0.3), seconds=2),
    "boat cockpit canopy": dict(
        group="mount", lens="ultra-wide",
        plate="camera mounted inside the enclosed canopy cockpit of {car} looking forward past the steering wheel and analog racing gauges, "
              "water spray sheeting across the reinforced canopy glass in {world}, {action}",
        motion="Canopy cockpit camera. The pilot's hands grip the racing wheel as the hull violently skips across waves, "
               "white spray exploding across the windshield, the rival boat locked door-to-door just feet away outside.",
        actions=["head-to-head drag", "buoy turn carve", "wave skip launch"],
        fx=dict(shake=0.7, speed_blur=0.6, zoom_punch=0.1, chroma=0.2), seconds=3),
    "boat transom wake": dict(
        group="mount", lens="wide",
        plate="reverse-facing camera mounted on the transom between the massive outboard racing engines of {car}, "
              "giant 50-foot rooster tails of white spray arching into the sky in {world}, {action}",
        motion="Reverse transom camera. Massive twin rooster tails of water shoot fifty feet into the air, water vapor clouding the wake, "
               "the open ocean tearing away behind at breakneck speed as engines scream at redline.",
        actions=["rooster tail blast", "open water sprint", "head-to-head drag"],
        fx=dict(shake=0.65, speed_blur=0.7, zoom_punch=0.0, chroma=0.25), seconds=3),
    "boat sponson side": dict(
        group="mount", lens="wide",
        plate="camera mounted on the outer hydroplane sponson of {car} looking forward along the sculpted fiberglass hull slicing through the water in {world}, {action}",
        motion="Sponson-mounted camera skimming the water's surface. The hull carves deep into a high-G turn around the buoy, "
               "a solid wall of ocean spray erupting off the chines, sunlight gleaming through the mist.",
        actions=["buoy turn carve", "wave skip launch", "rooster tail blast"],
        fx=dict(shake=0.75, speed_blur=0.7, zoom_punch=0.15, chroma=0.25), seconds=3),
}

GROUPS = {"mount": "vehicle-mounted", "chase": "chase / aerial", "trackside": "trackside"}
HOLD = {"mount": "The camera rig is bolted to the vehicle, so the vehicle bodywork stays locked in the same place in frame the whole time. "
                  "The camera never pulls back, never zooms out and never separates from the vehicle; the same part of the vehicle "
                  "fills the same part of the frame from the first frame to the last.",
        "chase": "The hero vehicle stays large and centered in the frame the whole time, filling at least a third of the frame; "
                 "the camera keeps pace at the same distance, never falls behind, never pulls back and never loses it.",
        "trackside": "The camera tracks the vehicles cleanly across the frame the whole time."}

# Curated fast-paced race arc: alternates car-mounted, chase/aerial, and trackside rigs along a real race story
RACE_SEQUENCE = ARCHETYPES["car"]["sequence"]

# 'auto (cycle by shot)' walks this order
AUTO_ORDER = ARCHETYPES["car"]["auto_order"]

SPEED = [
    (0.34, "fast", "", "engine humming"),
    (0.67, "very fast", "intense, ", "engine roaring"),
    (1.01, "insanely fast", "violent, aggressive, ", "twin race engines screaming at the redline"),
]

ASPECTS = {
    "16:9 landscape": (768, 448, 1024, 576),
    "9:16 vertical": (448, 768, 576, 1024),
    "2.35:1 cinema": (896, 384, 1152, 512),
}


def rig_for(rig: str, shot_index: int, seed: int, subject_type: str = "car") -> str:
    if rig in RIGS:
        return rig
    arch = ARCHETYPES.get(subject_type, ARCHETYPES["car"])
    seq = arch.get("sequence", RACE_SEQUENCE)
    if rig.startswith("auto (race"):
        return seq[shot_index % len(seq)][0]
    auto_order = arch.get("auto_order", AUTO_ORDER)
    if rig.startswith("auto"):
        start = seed % len(auto_order)
        return auto_order[(start + shot_index) % len(auto_order)]
    if rig.startswith("random"):
        return random.Random(seed * 1009 + shot_index).choice(sorted(RIGS))
    raise KeyError(f"unknown rig {rig!r}; one of {sorted(RIGS)}")


def action_for(rig: str, rig_name: str, action: str, shot_index: int, seed: int, subject_type: str = "car") -> str:
    if action in ACTIONS:
        return action
    arch = ARCHETYPES.get(subject_type, ARCHETYPES["car"])
    seq = arch.get("sequence", RACE_SEQUENCE)
    if rig.startswith("auto (race"):
        return seq[shot_index % len(seq)][1]
    rng = random.Random(seed * 7919 + shot_index)
    return rng.choice(RIGS[rig_name]["actions"])


def compose(car: str = "", world: str = "mountain pass", rig: str = "auto (race sequence)", *,
            subject_type: str = "car", style: str = "photoreal", rival: str = "", world_custom: str = "",
            light: str = "world default", action: str = "auto", intensity: float = 0.85,
            shot_index: int = 0, seed: int = 0, aspect: str = "16:9 landscape", seconds: float = 0,
            extra: str = "") -> dict:
    """Compile one shot. Returns prompts, sizes, the resolved rig/action and Velocity FX defaults."""
    arch = ARCHETYPES.get(subject_type, ARCHETYPES["car"])
    st = STYLES.get(style, STYLES["photoreal"])
    car = (car or "").strip() or arch["default_subject"]
    rival_raw = (rival or "").strip()
    if rival_raw.lower() in ("none", "solo", "off", "false"):
        rival_desc = ""
    else:
        rival_desc = rival_raw or arch["default_rival"]

    rig_name = rig_for(rig, shot_index, seed, subject_type)
    r = RIGS[rig_name]
    if world == "custom" or world_custom.strip():
        w = dict(scene=world_custom.strip() or WORLDS["mountain pass"]["scene"],
                 light="cinematic natural light", sound="engines roaring")
    elif world in WORLDS:
        w = WORLDS[world]
    else:
        w = dict(scene=world.strip(), light="cinematic natural light", sound="engines roaring")
    light_text = LIGHTS.get(light) or w["light"]
    act_name = action_for(rig, rig_name, action, shot_index, seed, subject_type)
    act = ACTIONS[act_name]
    speed, edge, engine = next((s, e, n) for t, s, e, n in SPEED if intensity < t)

    if rival_desc:
        act_plate = act["plate"].format(rival=rival_desc)
        act_motion = act["motion"].format(rival="the rival")
    else:
        act_plate = act["solo_plate"]
        act_motion = act["solo_motion"]

    cap = lambda t: t[:1].upper() + t[1:]
    plate = r["plate"].format(car=car, world=w["scene"], action=act_plate)
    no_brand = arch["no_brand"]
    plate_prompt = (f"{cap(plate)}. {st['plate_desc']}, {r['lens']} lens, in {w['scene']}, {act_plate}. "
                    f"Intense motion blur on the background and surface while the vehicle is tack-sharp. "
                    f"{cap(light_text)}. {no_brand}")
    if extra.strip():
        plate_prompt += " " + extra.strip()
    
    hold_rule = arch.get("hold_" + r["group"], HOLD[r["group"]])
    engine_sound = arch.get("sound") or engine
    motion_prompt = (f"{r['motion']} {act_motion}, {speed}, {st['motion_desc']}, through {w['scene']}. "
                     f"{cap(edge + 'high-energy cinematics')}, {light_text}. "
                     f"Sound: {engine_sound}, {w['sound']}, wind rush. "
                     f"{hold_rule} The vehicle keeps its exact shape, paint and details the whole time.")
    if extra.strip():
        motion_prompt += " " + extra.strip()

    plate_neg = build_negative_prompt(arch["plate_neg"], st["plate_neg"], style)
    video_neg = build_negative_prompt(arch["video_neg"], st["video_neg"], style)

    vw, vh, pw, ph = ASPECTS[aspect]
    secs = seconds or r["seconds"]
    fps = 24
    fx = dict(r["fx"])
    tweak = st.get("fx_tweak", {})
    fx = {k: round(v * tweak.get(k, 1.0) * (0.5 + 0.5 * intensity), 3) for k, v in fx.items()}
    return dict(rig=rig_name, group=GROUPS[r["group"]], action=act_name, car=car, rival=rival_desc,
                subject_type=subject_type, style=style,
                plate_prompt=plate_prompt, motion_prompt=motion_prompt, plate_negative=plate_neg,
                video_negative=video_neg, width=vw, height=vh, plate_width=pw, plate_height=ph,
                seconds=secs, fps=fps, frames=int(secs * fps) + 1, seed=seed * 1000 + shot_index,
                fx=fx, label=f"{shot_index + 1:02d} {rig_name} / {act_name}")


# ----------------------------------------------------------------------------------------------------- custom rig builder
CUSTOM_ANCHORS = {
    "front_splitter": dict(
        group="mount", name="front splitter mount",
        view="very low camera mounted directly on the front bumper splitter or nose of {car}, centimeters above the surface, looking straight ahead",
        motion="Skimming centimeters above the racing surface at extreme speed, lane markings and terrain seams whipping beneath the lens with violent vibration",
        default_shake=0.8, default_blur=0.75, default_punch=0.2, default_chroma=0.3,
        actions=["drafting", "overtake", "hairpin", "near miss"], seconds=3),
    "wheel_arch_or_fork": dict(
        group="mount", name="wheel arch / fork mount",
        view="low close-up camera clamped inside the front wheel arch or suspension fork of {car}: the spinning tire and glowing brake disc fill the side of frame, surface inches below",
        motion="Camera rigidly bolted beside the spinning tire and glowing brake caliper. The wheel spins in a fierce blur, surface tearing past inches below the lens",
        default_shake=0.75, default_blur=0.7, default_punch=0.0, default_chroma=0.25,
        actions=["hairpin", "hard brake", "carve a bend", "hard trail brake"], seconds=3),
        "pilot_face_cam": dict(
        group="mount", name="pilot face-cam (cockpit)",
        view="glare-shield camera mounted on top of the instrument panel glare-shield inside {car}, pointing directly back into the face and helmet of the pilot, framing the pilot from the chest up with the canopy and rushing world behind",
        motion="Camera locked onto the pilot face inside the cockpit. The pilot helmet, eyes, and flight gear stay centered in frame, tensing against G-forces as the vehicle maneuvers at extreme speed",
        default_shake=0.75, default_blur=0.5, default_punch=0.25, default_chroma=0.25,
        actions=["high-g strain", "canyon trench run", "barrel roll evade"], seconds=3),
    "over_shoulder_pov": dict(
        group="mount", name="over-the-shoulder POV",
        view="over-the-shoulder camera mounted behind the pilot right shoulder inside the cockpit of {car}, framing the helmet, hands on controls, and glowing HUD looking forward through the canopy",
        motion="Over-the-shoulder camera locked behind the pilot. The helmet and hands stay anchored in frame, micro-correcting controls as the vehicle slices through turns at extreme velocity",
        default_shake=0.6, default_blur=0.6, default_punch=0.15, default_chroma=0.2,
        actions=["canyon trench run", "combat formation", "mach punch"], seconds=3),
    "crash_snap_zoom": dict(
        group="chase", name="crash snap zoom to cockpit",
        view="dynamic Hollywood crash zoom snapping from an exterior wide shot of {car} instantly punching straight through the canopy glass into the pilot focused eyes",
        motion="Dynamic crash zoom! The camera snaps violently from a wide exterior formation shot straight through the bubble canopy glass to lock onto the pilot focused eyes",
        default_shake=0.85, default_blur=0.8, default_punch=0.85, default_chroma=0.4,
        actions=["mach punch", "canyon trench run"], seconds=2),
    "cockpit_pov": dict(
        group="mount", name="cockpit POV",
        view="interior POV from inside the cockpit or helmet of {car} looking forward over the steering wheel, handlebars or flight stick, dashboard and controls in the lower foreground",
        motion="Cockpit-bolted camera. The steering controls and dash stay fixed in frame while the horizon tilts and road/clouds smear past with intense speed",
        default_shake=0.6, default_blur=0.6, default_punch=0.1, default_chroma=0.2,
        actions=["drafting", "overtake", "canyon trench run", "high-speed tuck"], seconds=3),
    "side_door_or_fairing": dict(
        group="mount", name="side door / fairing mount",
        view="camera suction-mounted on the side flank or fairing of {car} looking forward past the mirror or winglet along the bodywork toward the road ahead",
        motion="Side-mounted camera locked to the bodywork. The vehicle flank stays rock solid on one side of frame while the world smears past at intense speed",
        default_shake=0.5, default_blur=0.6, default_punch=0.1, default_chroma=0.2,
        actions=["overtake", "near miss", "carve a bend", "flat out"], seconds=3),
    "roof_or_canopy": dict(
        group="mount", name="roof / canopy mount",
        view="camera mounted center on the roof or canopy spine of {car} looking forward over the hood and windshield",
        motion="Roof-mounted camera riding with the vehicle. The nose points aggressively into every turn as the horizon tilts and speed rushes toward the lens",
        default_shake=0.5, default_blur=0.55, default_punch=0.0, default_chroma=0.15,
        actions=["carve a bend", "overtake", "hairpin", "barrel roll evade"], seconds=3),
    "exhaust_or_thrust": dict(
        group="mount", name="exhaust / thrust macro",
        view="tight close-up macro of the exhaust pipes, diffuser or afterburner nozzles of {car}, glowing red-hot with exhaust flames",
        motion="Tight camera locked on the exhaust nozzles: bright flames burst from the pipes, heat shimmer warps the air, diffuser vibrating violently",
        default_shake=0.7, default_blur=0.5, default_punch=0.45, default_chroma=0.35,
        actions=["hard brake", "launch", "mach punch", "quickshifter pop"], seconds=2),
    "rear_reverse": dict(
        group="mount", name="rear bumper reverse",
        view="reverse-facing camera mounted on the rear bumper or tail cowl of {car} looking backward at the unspooling surface and chasing rival behind",
        motion="Reverse-facing camera. The surface pours out from under the vehicle at breakneck speed while the rival weaves aggressively in the slipstream behind",
        default_shake=0.65, default_blur=0.65, default_punch=0.0, default_chroma=0.25,
        actions=["lead defense", "hard brake", "drift", "slipstream pass"], seconds=3),
    "underbody_keel": dict(
        group="mount", name="underbody / keel mount",
        view="extreme ground-hugging camera mounted under the chassis or hull of {car} just millimeters above the rushing surface, sparks or water spray flying",
        motion="Underbody camera skimming millimeters above the ground. The rushing surface tears directly underneath the lens in a violent blur, sparks or spray showering outward",
        default_shake=0.85, default_blur=0.8, default_punch=0.2, default_chroma=0.35,
        actions=["launch", "flat out", "jump", "wave skip launch"], seconds=2),
    "wingtip_rail": dict(
        group="mount", name="wingtip / spoiler rail",
        view="aerodynamic pod camera bolted to the outer wingtip or rear spoiler endplate of {car} looking inward along the wing toward the cockpit",
        motion="Wingtip camera locked to the wing endplate. The wing surface leads across frame while the terrain or sky banks dizzyingly below",
        default_shake=0.55, default_blur=0.7, default_punch=0.1, default_chroma=0.2,
        actions=["afterburner climb", "canyon trench run", "barrel roll evade"], seconds=3),
    "chase_crane_low": dict(
        group="chase", name="chase crane arm (low)",
        view="low tracking shot from a gyro-stabilized crane arm on a chase vehicle, just off the rear quarter at bumper height",
        motion="Russian-arm crane matching speed off the rear quarter, holding low and swinging smoothly, background streaking into creamy horizontal blur while the hero stays sharp",
        default_shake=0.3, default_blur=0.55, default_punch=0.0, default_chroma=0.15,
        actions=["drafting", "overtake", "drift", "carve a bend"], seconds=3),
    "chase_orbit_sweep": dict(
        group="chase", name="chase crane orbit",
        view="dynamic sweeping tracking shot from a crane arm orbiting around {car} from rear three-quarter to front fender at speed",
        motion="The crane arm sweeps fast in an arc around the moving vehicle while matching extreme speed, background spinning into blur, hero centered",
        default_shake=0.25, default_blur=0.55, default_punch=0.0, default_chroma=0.15,
        actions=["overtake", "flat out", "carve a bend"], seconds=3),
    "fpv_drone_chase": dict(
        group="chase", name="FPV drone chase",
        view="FPV racing drone flying inches behind the rear wing or tail of {car}, banking hard through corners",
        motion="FPV drone locked inches behind the tail, banking, weaving and diving with the vehicle at breakneck speed through smoke and spray",
        default_shake=0.5, default_blur=0.65, default_punch=0.2, default_chroma=0.2,
        actions=["drift", "overtake", "near miss", "big gap jump"], seconds=3),
    "fpv_drone_dive": dict(
        group="chase", name="FPV drone dive",
        view="steep high-altitude FPV drone diving vertically toward the track and pulling out inches behind {car} into the apex",
        motion="FPV drone dives steeply toward the deck, pulling out into high-speed chase, horizon spinning dynamically",
        default_shake=0.5, default_blur=0.65, default_punch=0.25, default_chroma=0.2,
        actions=["drift", "hairpin", "berm railing", "overtake"], seconds=3),
    "overhead_drone_90deg": dict(
        group="chase", name="overhead drone 90°",
        view="bird's-eye view looking straight down from directly above, a 90-degree overhead tracking drone shot of {car}",
        motion="Overhead drone locked straight down, tracking the vehicle as it slices through corners, tire tracks and shadows streaming beneath",
        default_shake=0.15, default_blur=0.35, default_punch=0.0, default_chroma=0.05,
        actions=["drift", "hairpin", "overtake"], seconds=3),
    "roadside_whip_pan": dict(
        group="trackside", name="roadside whip pan",
        view="low roadside camera at the edge of the kerb, {car} charging head-on toward the lens at full throttle",
        motion="Low roadside camera. The vehicle roars past inches away and the camera whip-pans violently to follow it, background smearing into heavy streak blur",
        default_shake=0.7, default_blur=0.8, default_punch=0.15, default_chroma=0.3,
        actions=["finish flyby", "overtake", "near miss", "supersonic flyby"], seconds=2),
    "roadside_crash_zoom": dict(
        group="trackside", name="roadside crash zoom",
        view="low long telephoto shot looking straight down the road as {car} charges toward the camera through heat shimmer",
        motion="Telephoto camera snaps into a violent crash zoom onto the nose and headlights as the vehicle hurtles toward the lens",
        default_shake=0.55, default_blur=0.5, default_punch=1.0, default_chroma=0.4,
        actions=["hard brake", "overtake", "flat out", "finish drop"], seconds=2),
}

CUSTOM_PITCHES = {
    "level": ("level eye-level horizon", "the camera stays level with the horizon"),
    "tilt_up_hero": ("low-angle camera tilted upward at 20 degrees, making the vehicle look massive and imposing", "low camera tilted upward emphasizing height and aggressive scale"),
    "tilt_down_top": ("high-angle camera tilted downward at 25 degrees, emphasizing the track and road ahead", "tilted down toward the rushing terrain"),
    "dutch_tilt_canted": ("dutch-tilt canted angle tilted at 15 degrees, dynamic tilted horizon for extreme speed feel", "the camera is rolled at a canted Dutch angle with horizons tilted dynamically"),
    "reverse_looking_back": ("reverse-facing angle looking backward into the slipstream", "looking backward as the world unspools behind"),
}

CUSTOM_LENSES = {
    "12mm action fisheye": ("12mm ultra-wide action camera lens with extreme barrel distortion, wide field-of-view and heavy edge motion blur", 1.25, 0.35),
    "24mm cinematic wide": ("24mm wide-angle cinema lens, crisp natural perspective and edge speed blur", 1.0, 0.2),
    "35mm documentary": ("35mm documentary prime lens, true-to-life perspective, clean subject isolation", 0.8, 0.1),
    "50mm anamorphic": ("50mm anamorphic cinema lens, shallow depth-of-field, horizontal flare streak, oval bokeh", 0.7, 0.15),
    "85mm portrait telephoto": ("85mm fast telephoto lens, compressed background, tack-sharp vehicle against heavy bokeh blur", 0.6, 0.15),
    "135mm compressed telephoto": ("135mm long telephoto zoom lens, intense background compression and shimmering atmospheric heat distortion", 0.5, 0.3),
    "macro tight": ("100mm macro lens, ultra-shallow depth of field, mechanical details and surface textures filling the frame", 0.4, 0.3),
}

CUSTOM_STABILITY = {
    "rigid_bolt": ("rigidly bolted directly to the metal chassis, high-frequency mechanical engine vibration", 1.2),
    "suction_damped": ("suction-cup mounted to exterior bodywork with silicone dampeners, harmonic road oscillation", 1.0),
    "stabilized_gimbal": ("three-axis gyro-stabilized motorized gimbal mount, buttery smooth tracking with no jitter", 0.35),
    "handheld_raw": ("raw handheld cinema camera operated by a passenger or operator, natural human muscle micro-shake", 1.15),
}


def compose_custom_rig(*, subject_type: str = "car", style: str = "photoreal", mount_anchor: str = "front_splitter",
                       camera_pitch: str = "level", lens_profile: str = "24mm cinematic wide",
                       mount_stability: str = "rigid_bolt", action: str = "auto",
                       car: str = "", rival: str = "", world: str = "mountain pass", world_custom: str = "",
                       light: str = "world default", intensity: float = 0.85, shot_index: int = 0,
                       seed: int = 0, aspect: str = "16:9 landscape", seconds: float = 0.0, extra: str = "") -> dict:
    """Build a fully custom camera rig combining physical mount anchor, camera pitch, lens profile, mount stability, and animation style."""
    arch = ARCHETYPES.get(subject_type, ARCHETYPES["car"])
    st = STYLES.get(style, STYLES["photoreal"])
    car = (car or "").strip() or arch["default_subject"]
    rival_raw = (rival or "").strip()
    rival_desc = "" if rival_raw.lower() in ("none", "solo", "off", "false") else (rival_raw or arch["default_rival"])

    anchor = CUSTOM_ANCHORS.get(mount_anchor, CUSTOM_ANCHORS["front_splitter"])
    pitch = CUSTOM_PITCHES.get(camera_pitch, CUSTOM_PITCHES["level"])
    lens = CUSTOM_LENSES.get(lens_profile, CUSTOM_LENSES["24mm cinematic wide"])
    stability = CUSTOM_STABILITY.get(mount_stability, CUSTOM_STABILITY["rigid_bolt"])

    if world == "custom" or world_custom.strip():
        w = dict(scene=world_custom.strip() or WORLDS["mountain pass"]["scene"],
                 light="cinematic natural light", sound="engines roaring")
    elif world in WORLDS:
        w = WORLDS[world]
    else:
        w = dict(scene=world.strip(), light="cinematic natural light", sound="engines roaring")
    light_text = LIGHTS.get(light) or w["light"]

    if action in ACTIONS:
        act_name = action
    else:
        act_name = anchor["actions"][shot_index % len(anchor["actions"])]
    act = ACTIONS[act_name]
    speed, edge, engine = next((s, e, n) for t, s, e, n in SPEED if intensity < t)

    if rival_desc:
        act_plate = act["plate"].format(rival=rival_desc)
        act_motion = act["motion"].format(rival="the rival")
    else:
        act_plate = act["solo_plate"]
        act_motion = act["solo_motion"]

    cap = lambda t: t[:1].upper() + t[1:]
    view_text = anchor["view"].format(car=car)
    plate_prompt = (f"{cap(view_text)}, {pitch[0]}. {st['plate_desc']}, {lens[0]}, "
                    f"in {w['scene']}, {act_plate}. Intense motion blur on the background and surface while the vehicle is tack-sharp. "
                    f"{cap(light_text)}. {arch['no_brand']}")
    if extra.strip():
        plate_prompt += " " + extra.strip()

    hold_rule = arch.get("hold_" + anchor["group"], HOLD[anchor["group"]])
    engine_sound = arch.get("sound") or engine
    motion_prompt = (f"{anchor['motion']}, {pitch[1]}, {stability[0]}. {act_motion}, {speed}, {st['motion_desc']}, through {w['scene']}. "
                     f"{cap(edge + 'high-energy cinematography')}, {light_text}. Sound: {engine_sound}, {w['sound']}, wind rush. "
                     f"{hold_rule} The vehicle keeps its exact shape, paint and details the whole time.")
    if extra.strip():
        motion_prompt += " " + extra.strip()

    plate_neg = build_negative_prompt(arch["plate_neg"], st["plate_neg"], style)
    video_neg = build_negative_prompt(arch["video_neg"], st["video_neg"], style)

    vw, vh, pw, ph = ASPECTS[aspect]
    secs = seconds or anchor["seconds"]
    fps = 24

    tweak = st.get("fx_tweak", {})
    shake = round(anchor["default_shake"] * stability[1] * tweak.get("grain", 1.0) * (0.5 + 0.5 * intensity), 3)
    speed_blur = round(anchor["default_blur"] * lens[1] * tweak.get("speed_blur", 1.0) * (0.5 + 0.5 * intensity), 3)
    zoom_punch = round(anchor["default_punch"] * (0.5 + 0.5 * intensity), 3)
    chroma = round(anchor["default_chroma"] * (lens[2] / 0.2) * tweak.get("chroma", 1.0) * (0.5 + 0.5 * intensity), 3)
    fx = dict(shake=shake, speed_blur=speed_blur, zoom_punch=zoom_punch, chroma=chroma)

    label_clean = f"custom ({anchor['name']} | {camera_pitch} | {lens_profile.split()[0]} | {style})"
    return dict(rig=label_clean, group=GROUPS[anchor["group"]], action=act_name, car=car, rival=rival_desc,
                subject_type=subject_type, style=style,
                plate_prompt=plate_prompt, motion_prompt=motion_prompt, plate_negative=plate_neg,
                video_negative=video_neg, width=vw, height=vh, plate_width=pw, plate_height=ph,
                seconds=secs, fps=fps, frames=int(secs * fps) + 1, seed=seed * 1000 + shot_index,
                fx=fx, label=f"{shot_index + 1:02d} {label_clean} / {act_name}")


def shot_list(n: int, **kw) -> list[dict]:
    """n consecutive shots (rig 'auto (race sequence)' walks the race spine)."""
    kw.pop("shot_index", None)
    return [compose(shot_index=i, **kw) for i in range(n)]


def car_ref_prompt(car_desc: str, style_desc: str = "", airborne: bool = False) -> str:
    """Studio reference shot of one vehicle, used as the identity ref for Qwen Image Edit."""
    style_header = style_desc or "professional motorsport studio three-quarter front-and-side reference photograph"
    where = ("flying level in mid-air against a plain soft grey sky with the landing gear fully retracted"
             if airborne else "parked on a clean studio floor")
    return (f"Clean {style_header} of {car_desc}, {where} under soft diffused lighting so the full vehicle silhouette, paint finish, "
            f"racing number, aero components and wheel design are crystal clear. "
            f"Original fictional vehicle design with no real brand badges, no logos and no text.")
