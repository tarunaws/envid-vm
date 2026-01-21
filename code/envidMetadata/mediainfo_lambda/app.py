def handler(event, context):
    return {
        "ok": False,
        "error": "MediaInfo Lambda is disabled (AWS/S3 support removed)",
    }
