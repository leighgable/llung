pub const IMAGE_RESIZE_TOOL_SCHEMA: &str = r#"
{
    "name": "image_mask_resize",
    "description": "Resize and mask an image as a thumbnail",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "size": {"type": "integer", "default": 128},
            "mask": {"type": "boolean", "default": true},
            "radius": {"type": "integer", "default": 20},
        },
        "required": ["path"]
    }
}
"#;
