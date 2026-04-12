import json
import os
from ..utils.colored_print import color, style
script_directory = os.path.dirname(os.path.abspath(__file__))

default_dims = [
    { "label": "◼︎ (1:1) - 1024x1024", "value": "1024x1024" },
    { "label": "◼︎ (1:1) - 1280x1280", "value": "1280x1280" },
    { "label": "◼︎ (1:1) - 1536x1536", "value": "1536x1536" },
    { "label": "◼︎ (1:1) - 1792x1792", "value": "1792x1792" },
    { "label": "◼︎ (1:1) - 2048x2048", "value": "2048x2048" },

    { "label": "🁢 (9:16) - 720x1280",  "value": "720x1280"  },
    { "label": "🁢 (9:16) - 768x1344",  "value": "768x1344"  },
    { "label": "🁢 (9:16) - 1080x1920", "value": "1080x1920" },

    { "label": "🁢 (4:5) - 1024x1280", "value": "1024x1280" },
    { "label": "🁢 (4:5) - 1280x1600", "value": "1280x1600" },

    { "label": "🁢 (3:4) - 896x1152",  "value": "896x1152"  },

    { "label": "🁢 (2:3) - 960x1440",  "value": "960x1440"  },
    { "label": "🁢 (2:3) - 1216x1824", "value": "1216x1824" },

    { "label": "🀰 (16:9) - 1280x720",  "value": "1280x720"  },
    { "label": "🀰 (16:9) - 1344x768",  "value": "1344x768"  },
    { "label": "🀰 (16:9) - 1920x1080", "value": "1920x1080" },

    { "label": "🀰 (4:3) - 1152x896",  "value": "1152x896"  },

    { "label": "🀰 (5:4) - 1280x1024", "value": "1280x1024" },
    { "label": "🀰 (5:4) - 1600x1280", "value": "1600x1280" },

    { "label": "🀰 (3:2) - 1216x832",  "value": "1216x832"  },
    { "label": "🀰 (3:2) - 1440x960",  "value": "1440x960"  },
    { "label": "🀰 (3:2) - 1824x1216", "value": "1824x1216" },
]

sd15_dims = [
    { "label": "◼︎ square (1:1) - 512x512", "value": "512x512" },
    { "label": "◼︎ square (1:1) - 768x768", "value": "768x768" },
    { "label": "🀰 landscape (3:2) - 768x512",  "value": "768x512" },
    { "label": "🀰 landscape (16:9) - 912x512", "value": "912x512" },
    { "label": "🁢 portrait (2:3) - 512x768",  "value": "512x768" },
    { "label": "🁢 portrait (9:16) - 512x912", "value": "512x912" },
]

sdxl_dims = [
    { "label": "◼︎ square (1:1) - 1024x1024", "value": "1024x1024" },
    { "label": "◼︎ square (1:1) - 768x768",   "value": "768x768"   },
    { "label": "🀰 landscape (9:7) - 1152x896",  "value": "1152x896"  },
    { "label": "🀰 landscape (3:2) - 1216x832",  "value": "1216x832"  },
    { "label": "🀰 landscape (7:4) - 1344x768",  "value": "1344x768"  },
    { "label": "🀰 landscape (12:5) - 1536x640", "value": "1536x640"  },
    { "label": "🁢 portrait (7:9) - 896x1152",  "value": "896x1152"  },
    { "label": "🁢 portrait (2:3) - 832x1216",  "value": "832x1216"  },
    { "label": "🁢 portrait (4:7) - 768x1344",  "value": "768x1344"  },
    { "label": "🁢 portrait (5:12) - 640x1536", "value": "640x1536"  },
]

flux2_dims = [
    { "label": "◼︎ square (1:1) - 1024x1024", "value": "1024x1024" },
    { "label": "◼︎ square (1:1) - 1280x1280", "value": "1280x1280" },
    { "label": "◼︎ square (1:1) - 1536x1536", "value": "1536x1536" },
    { "label": "◼︎ square (1:1) - 1792x1792", "value": "1792x1792" },
    { "label": "◼︎ square (1:1) - 2048x2048", "value": "2048x2048" },

    { "label": "🀰 landscape (3:2) - 1152x768", "value": "1152x768" },
    { "label": "🀰 landscape (3:2) - 1440x960", "value": "1440x960" },
    { "label": "🀰 landscape (3:2) - 1536x1024", "value": "1536x1024" },
    { "label": "🀰 landscape (3:2) - 1920x1280", "value": "1920x1280" },

    { "label": "🁢 portrait (2:3) - 768x1152", "value": "768x1152" },
    { "label": "🁢 portrait (2:3) - 960x1440", "value": "960x1440" },
    { "label": "🁢 portrait (2:3) - 1024x1536", "value": "1024x1536" },
    { "label": "🁢 portrait (2:3) - 1280x1920", "value": "1280x1920" },

    { "label": "🀰 landscape (4:3) - 1024x768", "value": "1024x768" },
    { "label": "🀰 landscape (4:3) - 1280x960", "value": "1280x960" },
    { "label": "🀰 landscape (4:3) - 1600x1200", "value": "1600x1200" },
    { "label": "🀰 landscape (4:3) - 1792x1344", "value": "1792x1344" },
    { "label": "🀰 landscape (4:3) - 2048x1536", "value": "2048x1536" },

    { "label": "🁢 portrait (3:4) - 768x1024", "value": "768x1024" },
    { "label": "🁢 portrait (3:4) - 960x1280", "value": "960x1280" },
    { "label": "🁢 portrait (3:4) - 1200x1600", "value": "1200x1600" },
    { "label": "🁢 portrait (3:4) - 1344x1792", "value": "1344x1792" },
    { "label": "🁢 portrait (3:4) - 1536x2048", "value": "1536x2048" },

    { "label": "🀰 landscape (16:9) - 1024x576", "value": "1024x576" },
    { "label": "🀰 landscape (16:9) - 1280x720", "value": "1280x720" },
    { "label": "🀰 landscape (16:9) - 1600x896", "value": "1600x896" },
    { "label": "🀰 landscape (16:9) - 1920x1088", "value": "1920x1088" },

    { "label": "🁢 portrait (9:16) - 576x1024", "value": "576x1024" },
    { "label": "🁢 portrait (9:16) - 720x1280", "value": "720x1280" },
    { "label": "🁢 portrait (9:16) - 896x1600", "value": "896x1600" },
    { "label": "🁢 portrait (9:16) - 1088x1920", "value": "1088x1920" },

    { "label": "🀰 landscape (21:9) - 1024x432", "value": "1024x432" },
    { "label": "🀰 landscape (21:9) - 1344x576", "value": "1344x576" },
    { "label": "🀰 landscape (21:9) - 1680x720", "value": "1680x720" },
    { "label": "🀰 landscape (21:9) - 2016x864", "value": "2016x864" },

    { "label": "🁢 portrait (9:21) - 432x1008", "value": "432x1008" },
    { "label": "🁢 portrait (9:21) - 576x1344", "value": "576x1344" },
    { "label": "🁢 portrait (9:21) - 720x1680", "value": "720x1680" },
    { "label": "🁢 portrait (9:21) - 864x2016", "value": "864x2016" }
]

zimage_dims = [
    { "label": "◼︎ square (1:1) - 1024x1024",        "value": "1024x1024" },
    { "label": "◼︎ square (1:1) - 1152x1152",        "value": "1152x1152" },
    { "label": "◼︎ square (1:1) - 1280x1280",        "value": "1280x1280" },
    { "label": "◼︎ square (1:1) - 1408x1408",        "value": "1408x1408" },
    { "label": "◼︎ square (1:1) - 1536x1536",        "value": "1536x1536" },

    { "label": "🀰 landscape (3:2) - 960x640",       "value": "960x640"   },
    { "label": "🀰 landscape (3:2) - 1152x768",      "value": "1152x768"  },
    { "label": "🀰 landscape (3:2) - 1216x832",      "value": "1216x832"  },
    { "label": "🀰 landscape (3:2) - 1344x896",      "value": "1344x896"  },
    { "label": "🀰 landscape (3:2) - 1536x1024",     "value": "1536x1024" },

    { "label": "🁢 portrait (2:3) - 640x960",        "value": "640x960"   },
    { "label": "🁢 portrait (2:3) - 768x1152",       "value": "768x1152"  },
    { "label": "🁢 portrait (2:3) - 832x1216",       "value": "832x1216"  },
    { "label": "🁢 portrait (2:3) - 896x1344",       "value": "896x1344"  },
    { "label": "🁢 portrait (2:3) - 1024x1536",      "value": "1024x1536" },

    { "label": "🀰 landscape (16:9) - 1024x576",    "value": "1024x576"  },
    { "label": "🀰 landscape (16:9) - 1152x648",    "value": "1152x648"  },
    { "label": "🀰 landscape (16:9) - 1280x720",    "value": "1280x720"  },
    { "label": "🀰 landscape (16:9) - 1344x756",    "value": "1344x756"  },
    { "label": "🀰 landscape (16:9) - 1536x864",    "value": "1536x864"  },

    { "label": "🁢 portrait (9:16) - 576x1024",      "value": "576x1024"  },
    { "label": "🁢 portrait (9:16) - 648x1152",      "value": "648x1152"  },
    { "label": "🁢 portrait (9:16) - 720x1280",      "value": "720x1280"  },
    { "label": "🁢 portrait (9:16) - 756x1344",      "value": "756x1344"  },
    { "label": "🁢 portrait (9:16) - 864x1536",      "value": "864x1536"  },

    { "label": "🀰 landscape (21:9) - 1280x544",     "value": "1280x544"  },
    { "label": "🀰 landscape (21:9) - 1536x656",     "value": "1536x656"  }
]

qwen_image_dims = [
    { "label": "◼︎ square (1:1) - 1024x1024", "value": "1024x1024" },
    { "label": "◼︎ square hd (1:1) - 1328x1328", "value": "1328x1328" },
    { "label": "🀰 landscape (4:3) - 1472x1104", "value": "1472x1104" },
    { "label": "🀰 landscape (16:9) - 1664x928", "value": "1664x928" },
    { "label": "🁢 portrait (3:4) - 1104x1472", "value": "1104x1472" },
    { "label": "🁢 portrait (9:16) - 928x1664", "value": "928x1664" }
]



video_dims = [
    # SD
    { "label": "🀰 SD 480p (4:3) - 640x480",       "value": "640x480"   },
    { "label": "🀰 SD 480p (16:9) - 854x480",     "value": "854x480"  },

    # HD
    { "label": "🀰 HD 720p (16:9) - 1280x720",         "value": "1280x720"  },
    { "label": "🀰 HD 1080p (16:9) - 1920x1080",       "value": "1920x1080" },

    # 2K
    { "label": "🀰 2K DCI (17:9) - 2048x1080",         "value": "2048x1080" },
    { "label": "🀰 2K QHD (16:9) - 2560x1440",         "value": "2560x1440" },

    # UHD / 4K
    { "label": "🀰 4K UHD (16:9) - 3840x2160",         "value": "3840x2160" },
    { "label": "🀰 4K DCI (17:9) - 4096x2160",         "value": "4096x2160" },

    # Vertical (social / mobile)
    { "label": "🁢 vertical 480p (9:16) - 480x854",    "value": "480x854"   },
    { "label": "🁢 vertical 720p (9:16) - 720x1280",   "value": "720x1280"  },
    { "label": "🁢 vertical 1080p (9:16) - 1080x1920", "value": "1080x1920" },
    { "label": "🁢 vertical 1440p (9:16) - 1440x2560", "value": "1440x2560" },
    { "label": "🁢 vertical 4K (9:16) - 2160x3840",    "value": "2160x3840" },
]

class Y7Nodes_ImageSizePresets:

    # Class variable to store the loaded custom dimensions (None if not found)
    custom_dims = None
    custom_dims_loaded = False

    @classmethod
    def load_custom_dims(cls):
        """Try to load custom dimensions from JSON file once and cache the result."""
        if not cls.custom_dims_loaded:
            cls.custom_dims_loaded = True
            try:
                with open(os.path.join(script_directory, 'custom_dimensions.json')) as f:
                    cls.custom_dims = json.load(f)
            except FileNotFoundError:
                print(f"custom_dimensions.json not found in {script_directory}. Using defaults.", color.ORANGE)
                cls.custom_dims = None
            except json.JSONDecodeError:
                print(f"Error parsing custom_dimensions.json. Using defaults.", color.RED)
                cls.custom_dims = None
        return cls.custom_dims

    @classmethod
    def get_dims_for_preset(cls, preset_name):
        """Return the dimensions list for the given preset name."""
        if preset_name == "SD1.5":
            return sd15_dims
        elif preset_name == "SDXL":
            return sdxl_dims
        elif preset_name == "Flux.2":
            return flux2_dims
        elif preset_name == "Z-Image":
            return zimage_dims
        elif preset_name == "Qwen-Image":
            return qwen_image_dims
        elif preset_name == "Video":
            return video_dims
        elif preset_name == "Custom*":
            custom = cls.load_custom_dims()
            return custom if custom is not None else default_dims
        else:  # "default"
            return default_dims

    @classmethod
    def INPUT_TYPES(cls):
        custom = cls.load_custom_dims()

        # Build combined dimension list (deduplicated by label)
        # Order: default, flux2, qwen, custom (if available)
        seen_labels = set()
        combined_dims = []
        for dims in [default_dims, sd15_dims, sdxl_dims, flux2_dims, zimage_dims, qwen_image_dims, video_dims, custom or []]:
            for d in dims:
                if d["label"] not in seen_labels:
                    seen_labels.add(d["label"])
                    combined_dims.append(d)

        dimension_options = [d["label"] for d in combined_dims]
        dimension_options.append("Custom")

        return {
            "required": {
                "preset": (["Default", "SD1.5", "SDXL", "Flux.2", "Z-Image", "Qwen-Image", "Video", "Custom*"],),
                "dimension": (dimension_options,),
                "custom_w": ("INT", {
                        "default": 1024,
                        "min": 128,
                        "max": 7680,
                        "step": 16,
                        "tooltip": "custom width"
                    }),
                "custom_h": ("INT", {
                        "default": 1024,
                        "min": 128,
                        "max": 7680,
                        "step": 16,
                        "tooltip": "custom height"
                    }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "Y7Nodes/Utilss"

    def generate(self, preset, dimension, custom_w, custom_h):
        try:
            if dimension == "Custom":
                return (custom_w, custom_h)

            # Get dims for the selected preset
            dims = self.get_dims_for_preset(preset)

            # Find matching entry in the preset's set
            entry = next((d for d in dims if d["label"] == dimension), None)

            # Fallback: search all sets if not found in the preset's set
            if entry is None:
                all_dims = default_dims + sd15_dims + sdxl_dims + flux2_dims + zimage_dims + qwen_image_dims + video_dims + (self.load_custom_dims() or [])
                entry = next((d for d in all_dims if d["label"] == dimension), None)

            if entry:
                width, height = [int(x.strip()) for x in entry["value"].split('x')]
                return (width, height)
            else:
                print(f"Dimension not found: {dimension}", color.RED)
                return (1024, 1024)

        except Exception as e:
            print(f"Error processing dimension: {str(e)}", color.RED)
            return (1024, 1024)
