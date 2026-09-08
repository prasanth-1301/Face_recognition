import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis
from PIL import Image
import io


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Face Check-In / Check-Out",
    page_icon="👤",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

SIMILARITY_THRESHOLD = 0.60


# ============================================================
# LOAD INSIGHTFACE MODEL
# ============================================================

@st.cache_resource
def load_face_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    return model


face_model = load_face_model()


# ============================================================
# SESSION STATE
# ============================================================

if "active_people" not in st.session_state:
    st.session_state.active_people = []

if "completed_visits" not in st.session_state:
    st.session_state.completed_visits = []

if "next_id" not in st.session_state:
    st.session_state.next_id = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_face_embedding(image_bytes):
    """
    Convert uploaded camera image to an InsightFace embedding.

    Requires exactly one face in the image.
    """

    try:

        # Read image using Pillow
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Convert Pillow image to NumPy array
        img = np.array(image)

        # InsightFace expects BGR format.
        # Reverse RGB channels to BGR.
        img = img[:, :, ::-1].copy()

    except Exception as e:

        return (
            None,
            f"Could not read the image: {str(e)}"
        )

    try:

        faces = face_model.get(img)

    except Exception as e:

        return (
            None,
            f"Face detection failed: {str(e)}"
        )

    if len(faces) == 0:

        return (
            None,
            "No face detected. Please take another photo."
        )

    if len(faces) > 1:

        return (
            None,
            "Multiple faces detected. "
            "Please make sure only one person is in the camera."
        )

    face = faces[0]

    embedding = face.normed_embedding

    if embedding is None:

        return (
            None,
            "Could not generate a face embedding."
        )

    return embedding, None


def cosine_similarity(
    embedding1,
    embedding2
):
    """
    Calculate cosine similarity between two
    face embeddings.
    """

    embedding1 = np.asarray(
        embedding1
    )

    embedding2 = np.asarray(
        embedding2
    )

    norm1 = np.linalg.norm(
        embedding1
    )

    norm2 = np.linalg.norm(
        embedding2
    )

    if norm1 == 0 or norm2 == 0:

        return 0.0

    return float(

        np.dot(
            embedding1,
            embedding2
        )

        /

        (
            norm1 * norm2
        )
    )


def format_duration(total_seconds):
    """
    Convert seconds into a readable duration.
    """

    total_seconds = int(
        total_seconds
    )

    hours = (
        total_seconds // 3600
    )

    minutes = (
        total_seconds % 3600
    ) // 60

    seconds = (
        total_seconds % 60
    )

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m"
        )

    if minutes > 0:

        return (
            f"{minutes}m "
            f"{seconds}s"
        )

    return f"{seconds}s"


def find_best_match(embedding):
    """
    Compare checkout face against every
    currently checked-in person.
    """

    best_person = None

    best_similarity = -1

    for person in st.session_state.active_people:

        similarity = cosine_similarity(

            embedding,

            person[
                "embedding"
            ]
        )

        if similarity > best_similarity:

            best_similarity = similarity

            best_person = person

    return (
        best_person,
        best_similarity
    )


def create_active_dataframe():
    """
    Create dataframe showing people
    currently checked in.
    """

    rows = []

    now = datetime.now()

    for person in (
        st.session_state.active_people
    ):

        duration = (

            now -

            person[
                "checkin_time"
            ]

        ).total_seconds()

        rows.append({

            "Name":
                person[
                    "name"
                ],

            "Check-in":
                person[
                    "checkin_time"
                ].strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                ),

            "Time Inside":
                format_duration(
                    duration
                ),

            "Status":
                "Inside"
        })

    return pd.DataFrame(
        rows
    )


def create_history_dataframe():
    """
    Create dataframe showing completed visits.
    """

    rows = []

    for visit in (
        st.session_state.completed_visits
    ):

        rows.append({

            "Name":
                visit[
                    "name"
                ],

            "Check-in":
                visit[
                    "checkin_time"
                ].strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                ),

            "Check-out":
                visit[
                    "checkout_time"
                ].strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                ),

            "Time Inside":
                visit[
                    "duration"
                ],

            "Face Similarity":
                f"{visit['similarity']:.3f}"
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "👤 Face Check-In / Check-Out"
)

st.write(
    "Check in a person using their name and face. "
    "At checkout, the system recognizes the person automatically "
    "and calculates how long they stayed inside."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Settings"
)

st.sidebar.write(
    f"Face similarity threshold: "
    f"**{SIMILARITY_THRESHOLD:.2f}**"
)

st.sidebar.info(
    "The threshold is a starting value for the PoC. "
    "You should calibrate it using your actual camera and people."
)


# ============================================================
# TABS
# ============================================================

checkin_tab, checkout_tab, dashboard_tab = st.tabs(

    [

        "🟢 Check-In",

        "🔴 Check-Out",

        "📊 Dashboard"

    ]
)


# ============================================================
# CHECK-IN
# ============================================================

with checkin_tab:

    st.header(
        "Check-In"
    )

    st.write(
        "Enter the person's name and "
        "take one clear face photo."
    )

    name = st.text_input(

        "Person Name",

        placeholder="Enter name",

        key="checkin_name"
    )

    st.write(
        "Take Check-In Photo"
    )

    checkin_photo = st.camera_input(

        "Camera",

        key="checkin_camera"
    )

    if checkin_photo is not None:

        st.image(

            checkin_photo,

            caption="Check-in photo",

            width=300
        )

    checkin_button = st.button(

        "✅ Confirm Check-In",

        type="primary",

        use_container_width=True
    )

    if checkin_button:

        if not name.strip():

            st.error(
                "Please enter the person's name."
            )

        elif checkin_photo is None:

            st.error(
                "Please take a face photo."
            )

        else:

            with st.spinner(
                "Analyzing face..."
            ):

                embedding, error = (
                    get_face_embedding(
                        checkin_photo.getvalue()
                    )
                )

            if error:

                st.error(
                    error
                )

            else:

                # Check if same person
                # is already inside

                already_inside = False

                for person in (
                    st.session_state.active_people
                ):

                    similarity = (
                        cosine_similarity(

                            embedding,

                            person[
                                "embedding"
                            ]
                        )
                    )

                    if similarity >= (
                        SIMILARITY_THRESHOLD
                    ):

                        already_inside = True

                        st.warning(

                            f"{person['name']} "
                            "appears to already "
                            "be checked in."
                        )

                        break

                if not already_inside:

                    checkin_time = (
                        datetime.now()
                    )

                    person = {

                        "id":
                            st.session_state.next_id,

                        "name":
                            name.strip(),

                        "embedding":
                            embedding,

                        "checkin_time":
                            checkin_time,

                        "checkin_image":
                            checkin_photo.getvalue()
                    }

                    st.session_state.active_people.append(
                        person
                    )

                    st.session_state.next_id += 1

                    st.success(

                        f"✅ {name.strip()} "
                        "checked in successfully!"
                    )

                    st.info(

                        "Check-in time: "

                        f"{checkin_time.strftime('%I:%M:%S %p')}"
                    )


# ============================================================
# CHECK-OUT
# ============================================================

with checkout_tab:

    st.header(
        "Check-Out"
    )

    st.write(

        "Take a face photo. The system "
        "will automatically find the matching "
        "person among everyone currently inside."
    )

    if len(
        st.session_state.active_people
    ) == 0:

        st.warning(
            "No people are currently checked in."
        )

    else:

        st.write(

            "Currently inside: "

            f"**{len(st.session_state.active_people)} "
            "person(s)**"
        )

        checkout_photo = st.camera_input(

            "Checkout Camera",

            key="checkout_camera"
        )

        if checkout_photo is not None:

            st.image(

                checkout_photo,

                caption="Checkout photo",

                width=300
            )

        checkout_button = st.button(

            "🔴 Confirm Check-Out",

            type="primary",

            use_container_width=True
        )

        if checkout_button:

            if checkout_photo is None:

                st.error(
                    "Please take a checkout photo."
                )

            else:

                with st.spinner(
                    "Recognizing person..."
                ):

                    embedding, error = (
                        get_face_embedding(
                            checkout_photo.getvalue()
                        )
                    )

                if error:

                    st.error(
                        error
                    )

                else:

                    best_person, similarity = (
                        find_best_match(
                            embedding
                        )
                    )

                    if best_person is None:

                        st.error(

                            "No currently checked-in "
                            "person found."
                        )

                    elif similarity < (
                        SIMILARITY_THRESHOLD
                    ):

                        st.error(
                            "❌ Person not recognized."
                        )

                        st.write(

                            "Best similarity: "

                            f"**{similarity:.3f}**"
                        )

                        st.info(

                            "The face does not match any "
                            "currently checked-in person."
                        )

                    else:

                        checkout_time = (
                            datetime.now()
                        )

                        checkin_time = (
                            best_person[
                                "checkin_time"
                            ]
                        )

                        duration_seconds = (

                            checkout_time -

                            checkin_time

                        ).total_seconds()

                        duration = (
                            format_duration(
                                duration_seconds
                            )
                        )

                        completed_visit = {

                            "id":
                                best_person[
                                    "id"
                                ],

                            "name":
                                best_person[
                                    "name"
                                ],

                            "checkin_time":
                                checkin_time,

                            "checkout_time":
                                checkout_time,

                            "duration":
                                duration,

                            "duration_seconds":
                                duration_seconds,

                            "similarity":
                                similarity
                        }

                        st.session_state.completed_visits.append(
                            completed_visit
                        )

                        st.session_state.active_people = [

                            person

                            for person in (
                                st.session_state.active_people
                            )

                            if person[
                                "id"
                            ] != best_person[
                                "id"
                            ]
                        ]

                        st.success(

                            f"✅ {best_person['name']} "
                            "checked out successfully!"
                        )

                        st.markdown(

                            f"""
### ⏱️ Visit Summary

**Person:** {best_person['name']}

**Check-in:** {checkin_time.strftime('%I:%M:%S %p')}

**Check-out:** {checkout_time.strftime('%I:%M:%S %p')}

**Time Inside:** 🟢 **{duration}**

**Face Similarity:** {similarity:.3f}
"""
                        )


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    st.header(
        "📊 Attendance Dashboard"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(

            "Currently Inside",

            len(
                st.session_state.active_people
            )
        )

    with col2:

        st.metric(

            "Completed Visits",

            len(
                st.session_state.completed_visits
            )
        )

    with col3:

        total_visits = len(
            st.session_state.completed_visits
        )

        st.metric(

            "Total Visits",

            total_visits
        )

    st.divider()


    # --------------------------------------------------------
    # CURRENTLY INSIDE
    # --------------------------------------------------------

    st.subheader(
        "🟢 Currently Inside"
    )

    if len(
        st.session_state.active_people
    ) == 0:

        st.info(
            "Nobody is currently inside."
        )

    else:

        active_df = (
            create_active_dataframe()
        )

        st.dataframe(

            active_df,

            use_container_width=True,

            hide_index=True
        )


    st.divider()


    # --------------------------------------------------------
    # COMPLETED VISITS
    # --------------------------------------------------------

    st.subheader(
        "🔴 Completed Visits"
    )

    if len(
        st.session_state.completed_visits
    ) == 0:

        st.info(
            "No completed visits yet."
        )

    else:

        history_df = (
            create_history_dataframe()
        )

        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True
        )


    st.divider()


    # --------------------------------------------------------
    # CLEAR HISTORY
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Completed History",
        type="secondary"
    ):

        st.session_state.completed_visits = []

        st.success(
            "Completed visit history cleared."
        )

        st.rerun()