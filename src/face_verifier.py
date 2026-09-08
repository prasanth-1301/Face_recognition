import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceVerifier:

    def __init__(
        self,
        similarity_threshold=0.50,
        minimum_face_size=80,
        minimum_blur_score=20
    ):

        self.similarity_threshold = similarity_threshold

        self.minimum_face_size = minimum_face_size

        self.minimum_blur_score = minimum_blur_score


        print("Loading InsightFace model...")


        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )


        self.face_app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )


        print(
            "InsightFace model loaded successfully."
        )


    # =====================================================
    # FACE DETECTION
    # =====================================================

    def detect_single_face(
        self,
        image
    ):


        if image is None:

            return {
                "success": False,
                "message": "Invalid image.",
                "face": None,
                "cropped_face": None
            }


        faces = self.face_app.get(
            image
        )


        # -------------------------------------------------
        # NO FACE
        # -------------------------------------------------

        if len(faces) == 0:

            return {
                "success": False,
                "message": (
                    "No face detected. "
                    "Please make sure your face "
                    "is clearly visible."
                ),
                "face": None,
                "cropped_face": None
            }


        # -------------------------------------------------
        # MULTIPLE FACES
        # -------------------------------------------------

        if len(faces) > 1:

            return {
                "success": False,
                "message": (
                    f"Multiple faces detected "
                    f"({len(faces)}). "
                    "Please make sure only one "
                    "person is visible."
                ),
                "face": None,
                "cropped_face": None
            }


        # -------------------------------------------------
        # EXACTLY ONE FACE
        # -------------------------------------------------

        face = faces[0]


        bbox = face.bbox.astype(
            int
        )


        x1, y1, x2, y2 = bbox


        image_height, image_width = (
            image.shape[:2]
        )


        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            image_width,
            x2
        )

        y2 = min(
            image_height,
            y2
        )


        face_width = (
            x2 - x1
        )

        face_height = (
            y2 - y1
        )


        # -------------------------------------------------
        # FACE TOO SMALL
        # -------------------------------------------------

        if (
            face_width < self.minimum_face_size
            or
            face_height < self.minimum_face_size
        ):

            return {
                "success": False,
                "message": (
                    "Face is too small for reliable "
                    "verification. Please move closer "
                    "to the camera."
                ),
                "face": None,
                "cropped_face": None
            }


        # -------------------------------------------------
        # CROP FACE
        # -------------------------------------------------

        cropped_face = image[
            y1:y2,
            x1:x2
        ]


        if cropped_face.size == 0:

            return {
                "success": False,
                "message": (
                    "Could not crop detected face."
                ),
                "face": None,
                "cropped_face": None
            }


        return {
            "success": True,
            "message": (
                "Exactly one face detected."
            ),
            "face": face,
            "cropped_face": cropped_face
        }


    # =====================================================
    # BLUR SCORE
    # =====================================================

    def calculate_blur_score(
        self,
        image
    ):


        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()


        return float(
            blur_score
        )


    # =====================================================
    # QUALITY VALIDATION
    # =====================================================

    def validate_face_quality(
        self,
        cropped_face
    ):


        blur_score = (
            self.calculate_blur_score(
                cropped_face
            )
        )


        if (
            blur_score
            <
            self.minimum_blur_score
        ):

            return {
                "success": False,
                "message": (
                    "Image appears blurry. "
                    "Please capture a clearer photo."
                ),
                "blur_score": blur_score
            }


        return {
            "success": True,
            "message": (
                "Face quality is acceptable."
            ),
            "blur_score": blur_score
        }


    # =====================================================
    # PROCESS IMAGE
    # =====================================================

    def process_image(
        self,
        image
    ):


        face_result = (
            self.detect_single_face(
                image
            )
        )


        if not face_result["success"]:

            return {
                "success": False,
                "message": (
                    face_result["message"]
                ),
                "face": None,
                "cropped_face": None,
                "embedding": None,
                "blur_score": None
            }


        quality_result = (
            self.validate_face_quality(
                face_result[
                    "cropped_face"
                ]
            )
        )


        if not quality_result["success"]:

            return {
                "success": False,
                "message": (
                    quality_result[
                        "message"
                    ]
                ),
                "face": (
                    face_result["face"]
                ),
                "cropped_face": (
                    face_result[
                        "cropped_face"
                    ]
                ),
                "embedding": None,
                "blur_score": (
                    quality_result[
                        "blur_score"
                    ]
                )
            }


        embedding = (
            face_result[
                "face"
            ].embedding
        )


        return {
            "success": True,
            "message": (
                "Face processed successfully."
            ),
            "face": (
                face_result[
                    "face"
                ]
            ),
            "cropped_face": (
                face_result[
                    "cropped_face"
                ]
            ),
            "embedding": embedding,
            "blur_score": (
                quality_result[
                    "blur_score"
                ]
            )
        }


    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    @staticmethod
    def calculate_similarity(
        embedding1,
        embedding2
    ):


        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )


        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )


        embedding1 = (
            embedding1
            /
            np.linalg.norm(
                embedding1
            )
        )


        embedding2 = (
            embedding2
            /
            np.linalg.norm(
                embedding2
            )
        )


        similarity = np.dot(
            embedding1,
            embedding2
        )


        return float(
            similarity
        )


    # =====================================================
    # FIND BEST MATCH
    # =====================================================

    def find_best_match(
        self,
        checkout_embedding,
        people
    ):


        if len(people) == 0:

            return {
                "success": False,
                "message": (
                    "No people are currently "
                    "checked in."
                )
            }


        best_match = None

        best_similarity = -1


        for person in people:


            stored_embedding = np.asarray(
                person["embedding"],
                dtype=np.float32
            )


            similarity = (
                self.calculate_similarity(
                    checkout_embedding,
                    stored_embedding
                )
            )


            if (
                similarity
                >
                best_similarity
            ):

                best_similarity = similarity

                best_match = person


        # -------------------------------------------------
        # THRESHOLD CHECK
        # -------------------------------------------------

        is_match = (
            best_similarity
            >=
            self.similarity_threshold
        )


        return {
            "success": True,

            "is_match": is_match,

            "person": best_match,

            "similarity": best_similarity,

            "threshold": (
                self.similarity_threshold
            )
        }