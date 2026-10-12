"""Upload / download routes for message attachments."""

from __future__ import annotations

import base64
import binascii
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import uploads

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


class UploadBody(BaseModel):
    name: str
    data_base64: str


@router.post("", status_code=201)
def upload(body: UploadBody):
    """JSON upload (base64) so no multipart dependency is needed."""
    try:
        data = base64.b64decode(body.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 data") from exc
    try:
        return uploads.save_upload(body.name, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{att_id}")
def download(att_id: str):
    meta = uploads.get_meta(att_id)
    path = uploads.file_path(att_id)
    if meta is None or path is None:
        raise HTTPException(status_code=404, detail="file not found")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", meta["name"]) or "file"
    return FileResponse(
        path,
        media_type=meta["mime"],
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{ascii_name}"',
            "Cache-Control": "private, max-age=3600",
        },
    )
