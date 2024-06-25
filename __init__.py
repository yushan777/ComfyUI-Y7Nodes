# combine node mappings from various modules into single dictionaries, which are then imported and used by other parts of the application or by users of this package

from .Y7TokenCounter import NODE_CLASS_MAPPINGS as NodeMappings, NODE_DISPLAY_NAME_MAPPINGS as NodeDisplayNames
from .Y7PhotoPrompter import NODE_CLASS_MAPPINGS as PhotoMappings, NODE_DISPLAY_NAME_MAPPINGS as PhotoDisplayNames
from .Y7ArtStylePrompter import NODE_CLASS_MAPPINGS as ArtStyleMappings, NODE_DISPLAY_NAME_MAPPINGS as ArtStyleDisplayNames

NODE_CLASS_MAPPINGS = {**NodeMappings, **PhotoMappings, **ArtStyleMappings}
NODE_DISPLAY_NAME_MAPPINGS = {**NodeDisplayNames, **PhotoDisplayNames, **ArtStyleDisplayNames}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


