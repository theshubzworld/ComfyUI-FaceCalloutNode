
from .FaceCalloutNode import FaceCalloutNode
from .IsolatedFaceCalloutNode import IsolatedFaceCalloutNode
from .IntegratedFaceCompositeNode import IntegratedFaceCompositeNode

NODE_CLASS_MAPPINGS = {
    "FaceCalloutEffect": FaceCalloutNode,
    "IsolatedFaceCallout": IsolatedFaceCalloutNode,
    "IntegratedFaceComposite": IntegratedFaceCompositeNode,
     
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCalloutEffect": "Face Callout Effect ✨",
    "IsolatedFaceCallout": "Isolated Face Callout ✂️",
    "IntegratedFaceComposite": "Integrated Face Composite 🎭" 
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- Loading Custom Callout Nodes ---")
print("  - Face Callout Effect")
print("  - Isolated Face Callout")
print("  - Integrated Face Composite") 
print("------------------------------------")