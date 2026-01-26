# Envid Metadata Service (DEPRECATED)

> **Deprecated:** This folder is the legacy AWS-based Envid Metadata service and is no longer part of the default (slim) stack.
>
> Use the multimodal metadata service instead:
> - Envid metadata service (multimodal) (port 5016): `microservices/backend/code/backend.py` (backend/orchestrator entrypoint)
>
> See: `microservices/backend/code/backend.py`.

## Overview
Extract metadata from videos (tags, transcript, emotions, on-screen text, celebrities).

Note: semantic/vector search has been removed; `/search` and `/search-text` return `410 Gone`.

This README is retained for historical reference. The current local-first/GCS-first implementation lives under `microservices/backend/code/backend.py`.

## Features

### 🎥 Video Analysis
- **Visual Recognition**: Extracts objects, scenes, activities using Amazon Rekognition
- **Speech-to-Text**: Transcribes audio content using Amazon Transcribe
- **Text Detection**: Identifies on-screen text in video frames
- **Emotion Detection**: Recognizes emotions from faces in the video

### 🔍 Search Capabilities
Removed.

## Architecture

```
Video Upload → Frame Extraction → Rekognition Analysis → Audio Transcription
                                           ↓
                                   Metadata Aggregation
             ↓
        JSON + Artifact Storage
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Upload Video
```bash
POST /upload-video
Content-Type: multipart/form-data

Fields:
- video: Video file (mp4, mov, avi, etc.)
- title: Video title (optional)
- description: Video description (optional)

Response:
{
  "id": "uuid",
  "title": "My Video",
  "message": "Video uploaded and indexed successfully",
  "labels_count": 45,
  "transcript_length": 1234,
  "frame_count": 12
}
```

### 3. Search Videos
```bash
POST /search
Response: `410 Gone`
{
  "error": "Search has been removed"
}
```

### 4. List All Videos
```bash
GET /videos

Response:
{
  "videos": [...],
  "total": 10
}
```

### 5. Get Video Details
```bash
GET /video/<video_id>

Response:
{
  "id": "uuid",
  "title": "My Video",
  "description": "...",
  "transcript": "Full transcript...",
  "labels": ["Label1", "Label2", ...],
  "text_detected": ["Text1", "Text2", ...],
  "emotions": ["HAPPY", "CALM"],
  "thumbnail": "base64_encoded_image",
  "uploaded_at": "2025-10-23T..."
}
```

## AWS Services Used

### Amazon Rekognition
- **DetectLabels**: Identifies objects, scenes, and activities
- **DetectText**: Extracts text from video frames
- **DetectFaces**: Analyzes faces and emotions

### Amazon Transcribe
- **StartTranscriptionJob**: Converts audio to text
- Supports multiple languages
- Provides timestamps for each word

### Amazon Bedrock
- **Titan Embeddings**: Generates 1536-dimensional vectors
- Enables semantic similarity matching
- Fast inference for real-time search

### Amazon S3
- Temporary storage for transcription jobs
- Stores audio files for processing

## Setup

### Prerequisites
- Python 3.8+
- FFmpeg installed
- AWS credentials configured
- AWS services enabled:
  - Rekognition
  - Transcribe
  - Bedrock (Titan Embeddings model)
  - S3

### Environment Variables
```bash
AWS_REGION=us-east-1
SEMANTIC_SEARCH_BUCKET=mediagenai-semantic-search
FFMPEG_PATH=/opt/homebrew/bin/ffmpeg  # Optional

# Performance tuning (optional)
ENVID_METADATA_PORT=5014  # legacy
ENVID_METADATA_FRAME_WORKERS=4
ENVID_METADATA_MAX_CELEBRITY_FRAMES=8
ENVID_METADATA_MAX_LABEL_FRAMES=6
ENVID_METADATA_MAX_TEXT_FRAMES=4
ENVID_METADATA_MAX_MODERATION_FRAMES=4
ENVID_METADATA_MAX_FACE_FRAMES=0  # 0 = run face detection on all frames

# Store original uploaded videos in S3 (default: enabled)
ENVID_METADATA_UPLOAD_VIDEOS_TO_S3=true
ENVID_METADATA_VIDEO_BUCKET=mediagenailab  # defaults to SEMANTIC_SEARCH_BUCKET/MEDIA_S3_BUCKET when unset
ENVID_METADATA_VIDEO_S3_PREFIX=envid-metadata/videos
```

### Installation
```bash
cd envidMetadata
pip install -r requirements.txt
python envidMetadata.py
```

Legacy service runs on **port 5014** (configurable via `ENVID_METADATA_PORT`).

## How It Works

### Video Indexing Process

1. **Frame Extraction**
   - Extracts frames at an interval (defaults are duration-aware; short videos use denser sampling)
   - Converts to JPEG for analysis
   - Limits frames analyzed via request params (and internal caps) to control API usage

2. **Visual Analysis**
   - Analyzes frames with Rekognition (labels/text/faces/moderation)
   - Uses sampling + bounded parallelism to reduce processing time
   - Runs celebrity detection on a capped subset of face-containing frames
   - Aggregates results across frames

3. **Audio Transcription**
   - Extracts audio as MP3 (16kHz mono)
   - Uploads to S3 for Transcribe
   - Downloads and parses transcript
   - Cleans up temporary files

4. **Metadata Compilation**
   - Combines: title, description, transcript, labels, text, emotions
   - Creates comprehensive text representation
   - Example: "Beach Day Family vacation Transcript: Look at the waves... Visual elements: Beach, Ocean, Child, Sunset Emotions: HAPPY, CALM"

5. **Embedding Generation**
   - Sends metadata to Bedrock Titan Embeddings
   - Receives 1536-dimensional vector
   - Vector captures semantic meaning

6. **Indexing**
   - Stores video entry with embedding
   - Includes thumbnail (middle frame as base64)
   - Maintains in-memory index for fast lookup

### Search Process

1. **Query Processing**
   - User enters natural language query
   - Example: "happy moments at sunset"

2. **Query Embedding**
   - Converts query to same embedding space
   - Uses Bedrock Titan Embeddings

3. **Similarity Calculation**
   - Computes cosine similarity between query and all videos
   - Formula: `similarity = dot(query, video) / (||query|| * ||video||)`
   - Score ranges from 0 (completely different) to 1 (identical)

4. **Ranking**
   - Sorts videos by similarity score
   - Returns top K results (default: 5)
   - Includes metadata and thumbnail

## Search Examples

### Scene-Based Search
```json
{
  "query": "sunset on the beach",
  "top_k": 3
}
```
Finds videos with beach scenes, ocean, and sunset lighting.

### Dialogue-Based Search
```json
{
  "query": "thank you speech",
  "top_k": 5
}
```
Finds videos where someone says "thank you" in audio.

### Emotion-Based Search
```json
{
  "query": "happy celebration moments",
  "top_k": 5
}
```
Finds videos with joyful emotions and celebratory scenes.

### Activity-Based Search
```json
{
  "query": "people dancing",
  "top_k": 5
}
```
Finds videos with dancing activity detected.

### Text-Based Search
```json
{
  "query": "news broadcast with headlines",
  "top_k": 3
}
```
Finds videos with on-screen text and news-related content.

## Performance Considerations

### API Call Optimization
- Limits frame analysis to 30 frames per video
- Extracts frames every 10 seconds (not every frame)
- Uses batch processing where possible

### Processing Time
- Small video (1-2 min): ~2-3 minutes
- Medium video (5-10 min): ~5-8 minutes
- Large video (20+ min): ~10-15 minutes

Bottlenecks:
- Frame extraction: Fast (seconds)
- Rekognition analysis: Moderate (30 frames × 3 API calls)
- Transcription: Slow (real-time or slower)
- Embedding generation: Fast (< 1 second)

### Cost Optimization
- Rekognition: $0.001 per image (DetectLabels, DetectText, DetectFaces)
  - 30 frames × 3 APIs = $0.09 per video
- Transcribe: $0.0004 per second
  - 10-minute video = $0.24
- Bedrock Titan Embeddings: $0.0001 per 1K tokens
  - ~$0.001 per video
- S3 storage: Minimal (temporary files deleted)

**Total cost per video: ~$0.33**

## Production Recommendations

### Use OpenSearch
Replace in-memory index with **Amazon OpenSearch Service**:
```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "your-domain.us-east-1.es.amazonaws.com", "port": 443}],
    use_ssl=True
)

# Index video
client.index(
    index="videos",
    body={
        "title": video_title,
        "embedding": embedding,
        "metadata": metadata
    }
)

# Search with k-NN
results = client.search(
    index="videos",
    body={
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 5
                }
            }
        }
    }
)
```

### Persistent Storage
- Store video metadata in **DynamoDB**
- Store actual videos in **S3**
- Use **CloudFront** for video delivery

### Async Processing
- Use **SQS + Lambda** for video processing
- Return job ID immediately
- Poll for completion status
- Enable processing of multiple videos in parallel

### Caching
- Cache embeddings in **ElastiCache**
- Cache frequent queries
- Reduce Bedrock API calls

## Limitations

### Current Implementation
- **In-memory index**: Lost on restart (use OpenSearch for production)
- **No video storage**: Videos not saved (only metadata)
- **Synchronous processing**: Blocks during upload (use async for production)
- **Single region**: No multi-region support

### AWS Service Limits
- Rekognition: 5 TPS per API
- Transcribe: 100 concurrent jobs
- Bedrock: 200 requests/minute (Titan Embeddings)

## Future Enhancements

1. **Video Segmentation**
   - Split long videos into scenes
   - Search within specific timestamps
   - Return video clips instead of full videos

2. **Multi-Modal Search**
   - Upload image to find similar scenes
   - Audio-based search (find videos with similar music)

3. **Advanced Filters**
   - Filter by duration, upload date, specific labels
   - Combine semantic + keyword search

4. **Real-Time Processing**
   - Stream processing for live videos
   - Incremental indexing

5. **Personalization**
   - User-specific rankings
   - Search history and preferences

## Troubleshooting

### FFmpeg Not Found
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

### S3 Bucket Access Denied
Ensure IAM role/user has:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:CreateBucket",
    "s3:PutObject",
    "s3:GetObject",
    "s3:DeleteObject"
  ],
  "Resource": "arn:aws:s3:::mediagenai-semantic-search/*"
}
```

### Bedrock Model Not Accessible
Enable Titan Embeddings in Bedrock console:
1. Go to AWS Bedrock console
2. Click "Model access"
3. Enable "Titan Embeddings G1 - Text"
4. Wait for approval (usually instant)

### Transcription Job Timeout
Increase `max_wait` in `_extract_audio_and_transcribe()` for longer videos.

## License
Part of MediaGenAI platform.
