#!/usr/bin/env python3

"""Deprecated.

This repository no longer supports the EC2/SQS-based ffmpeg worker. Proxy transcoding is now
performed locally by the metadata application host (ffmpeg is a prerequisite on that host).
"""

import sys

sys.stderr.write(
    "This script is deprecated: ffmpeg proxy transcoding is performed locally by the envidMetadata app.\n"
)
sys.exit(2)
