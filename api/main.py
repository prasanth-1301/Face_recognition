from fastapi import FastAPI, UploadFile, File, Form, HTTPException

import cv2
import numpy as np

from src.face_verifier import FaceVerifier
from src.database import AttendanceDatabase


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Face Verification API",
    description="Local InsightFace / ArcFace Face Verification Service",
    version="1.0.0"
)


# =========================================================
# LOAD FACE VERIFICATION MODEL
# =========================================================

face_verifier = FaceVerifier(
    similarity_threshold=0.50,
    minimum_face_size=80,
    minimum_blur_score=20
)


# =========================================================
# LOAD DATABASE
# =========================================================

database = AttendanceDatabase(
    db_path="data/attendance.db"
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Face Verification API is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# IMAGE CONVERSION
# =========================================================

def upload_to_cv2_image(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    return image


# =========================================================
# CHECK-IN API
# =========================================================

@app.post("/api/checkin")
async def check_in(
    partner_id: str = Form(...),
    image: UploadFile = File(...)
):

    # -----------------------------------------------------
    # VALIDATE FILE TYPE
    # -----------------------------------------------------

    if not image.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )


    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_bytes = await image.read()

    cv2_image = upload_to_cv2_image(
        image_bytes
    )


    if cv2_image is None:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )


    # -----------------------------------------------------
    # PROCESS FACE
    # -----------------------------------------------------

    result = face_verifier.process_image(
        cv2_image
    )


    if not result["success"]:

        raise HTTPException(
            status_code=400,
            detail=result["message"]
        )


    # -----------------------------------------------------
    # PREVENT DUPLICATE CHECK-IN
    # -----------------------------------------------------

    if database.is_name_checked_in(
        partner_id
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Partner {partner_id} "
                "is already checked in."
            )
        )


    # -----------------------------------------------------
    # SAVE CHECK-IN
    # -----------------------------------------------------

    record_id = database.check_in_person(
        name=partner_id,
        embedding=result["embedding"]
    )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "success": True,
        "message": "Check-in successful",
        "partner_id": partner_id,
        "checkin_id": record_id,
        "blur_score": round(
            result["blur_score"],
            2
        )
    }


# =========================================================
# CHECK-OUT API
# =========================================================

@app.post("/api/checkout")
async def check_out(
    image: UploadFile = File(...)
):

    # -----------------------------------------------------
    # VALIDATE FILE TYPE
    # -----------------------------------------------------

    if not image.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )


    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_bytes = await image.read()

    cv2_image = upload_to_cv2_image(
        image_bytes
    )


    if cv2_image is None:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )


    # -----------------------------------------------------
    # PROCESS CHECKOUT FACE
    # -----------------------------------------------------

    result = face_verifier.process_image(
        cv2_image
    )


    if not result["success"]:

        raise HTTPException(
            status_code=400,
            detail=result["message"]
        )


    # -----------------------------------------------------
    # GET ACTIVE CHECKED-IN PEOPLE
    # -----------------------------------------------------

    active_people = (
        database.get_active_people_with_embeddings()
    )


    if len(active_people) == 0:

        raise HTTPException(
            status_code=400,
            detail="No people are currently checked in."
        )


    # -----------------------------------------------------
    # FIND BEST FACE MATCH
    # -----------------------------------------------------

    match_result = face_verifier.find_best_match(
        checkout_embedding=result["embedding"],
        people=active_people
    )


    if not match_result["success"]:

        raise HTTPException(
            status_code=400,
            detail=match_result["message"]
        )


    # -----------------------------------------------------
    # NO MATCH
    # -----------------------------------------------------

    if not match_result["is_match"]:

        return {
            "success": True,
            "verified": False,
            "result": "DIFFERENT PERSON",
            "message": (
                "No matching checked-in person found."
            ),
            "similarity": round(
                match_result["similarity"],
                4
            ),
            "threshold": match_result["threshold"]
        }


    # -----------------------------------------------------
    # MATCH FOUND
    # -----------------------------------------------------

    matched_person = (
        match_result["person"]
    )


    checkout_result = (
        database.check_out_person(
            record_id=matched_person["id"],
            similarity=match_result["similarity"]
        )
    )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "success": True,

        "verified": True,

        "result": "SAME PERSON",

        "message": (
            "Face verified and checkout completed."
        ),

        "partner_id": (
            matched_person["name"]
        ),

        "similarity": round(
            match_result["similarity"],
            4
        ),

        "threshold": (
            match_result["threshold"]
        ),

        "checkin_time": (
            checkout_result[
                "checkin_time"
            ].isoformat()
        ),

        "checkout_time": (
            checkout_result[
                "checkout_time"
            ].isoformat()
        ),

        "duration_seconds": (
            checkout_result[
                "duration_seconds"
            ]
        )
    }