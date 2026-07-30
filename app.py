import streamlit as st
from dotenv import load_dotenv

# Load GEMINI_API_KEY (and any other vars) from a local .env file so the
# Gemini SDK can authenticate. Does nothing if there's no .env file.
load_dotenv()

import rag
from pawpal_system import Owner, Pet, Scheduler, Task


@st.cache_resource
def get_knowledge_base() -> rag.KnowledgeBase:
    """Build the Ask PawPal knowledge base once and reuse it across reruns.

    Cached so the vetted docs are read and BM25-indexed a single time, not on
    every Streamlit rerun.
    """
    return rag.KnowledgeBase()

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Plan a day of pet care. Add your pets and their tasks, set how much time you
have today, and PawPal+ builds a prioritized schedule and explains its choices.
"""
)

# --- Session "vault": a registry of owners, each with their own pets --------
# Keyed by name so switching owners swaps in that owner's pets/tasks instead
# of relabelling one shared Owner.
if "owners" not in st.session_state:
    st.session_state.owners = {}  # name -> Owner
if "current_owner" not in st.session_state:
    st.session_state.current_owner = None

st.divider()

# --- Owner -------------------------------------------------------------------
st.subheader("Owner")

col_pick, col_new = st.columns(2)
with col_pick:
    known = sorted(st.session_state.owners)
    if known:
        current = st.session_state.current_owner
        idx = known.index(current) if current in known else 0
        selected = st.selectbox("Choose owner", known, index=idx)
        st.session_state.current_owner = selected
    else:
        st.caption("No owners yet — add one on the right. →")
with col_new:
    new_owner_name = st.text_input("Add a new owner", value="")
    if st.button("Add owner"):
        name = new_owner_name.strip()
        if not name:
            st.warning("Enter a name for the new owner.")
        elif name in st.session_state.owners:
            st.warning(f"{name} already exists — pick them on the left.")
        else:
            st.session_state.owners[name] = Owner(name)
            st.session_state.current_owner = name
            st.success(f"Added owner {name}.")

owner = st.session_state.owners.get(st.session_state.current_owner)
if owner is None:
    st.info("Add an owner above to get started.")
    st.stop()

st.caption(f"Editing pets and tasks for **{owner.name}**.")

st.divider()

# --- Pets --------------------------------------------------------------------
st.subheader("Pets")

col1, col2, col3 = st.columns(3)
with col1:
    pet_name = st.text_input("Pet name", value="Mochi")
with col2:
    species = st.selectbox("Species", ["dog", "cat", "other"])
with col3:
    age = st.number_input("Age", min_value=0, max_value=40, value=2)

if st.button("Add pet"):
    existing_names = [pet.name for pet in owner.pet_list]
    if not pet_name.strip():
        st.warning("Give the pet a name first.")
    elif pet_name in existing_names:
        st.warning(f"{pet_name} is already in the list.")
    else:
        owner.add_pet(Pet(pet_name, species, int(age)))
        st.success(f"Added {pet_name}.")

if owner.pet_list:
    st.write("Current pets:")
    st.table(
        [
            {"name": pet.name, "species": pet.animal_type, "age": pet.age,
             "tasks": len(pet.tasks)}
            for pet in owner.pet_list
        ]
    )
else:
    st.info("No pets yet. Add one above.")

st.divider()

# --- Tasks -------------------------------------------------------------------
st.subheader("Tasks")

if not owner.pet_list:
    st.info("Add a pet before adding tasks.")
else:
    pet_names = [pet.name for pet in owner.pet_list]
    target_pet_name = st.selectbox("Add task to", pet_names)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        task_title = st.text_input("Task title", value="Morning walk")
    with col2:
        category = st.selectbox(
            "Category", ["walk", "feeding", "meds", "enrichment", "grooming", "other"]
        )
    with col3:
        duration = st.number_input(
            "Duration (min)", min_value=1, max_value=240, value=20
        )
    with col4:
        priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)
    with col5:
        repeat = st.selectbox("Repeat", ["none", "daily", "weekly"])

    fixed_time = st.text_input(
        "Fixed start time (HH:MM, optional)",
        value="",
        help="Pin this task to a set time, e.g. 08:00. Leave blank to let "
        "PawPal+ place it automatically. A blank or invalid time is treated "
        "as flexible.",
    )

    if st.button("Add task"):
        target_pet = next(p for p in owner.pet_list if p.name == target_pet_name)
        recurrence = None if repeat == "none" else repeat
        target_pet.add_task(
            Task(
                task_title,
                category,
                int(duration),
                priority,
                recurrence=recurrence,
                preferred_time=fixed_time.strip() or None,
            )
        )
        st.success(f"Added '{task_title}' to {target_pet_name}.")

    # Show every pet's tasks, sorted the way the scheduler would consider them
    # (high priority first, then shortest) so the ordering the plan uses is
    # visible here too. time_available is irrelevant to sorting, so 0 is fine.
    view_scheduler = Scheduler(owner, 0)
    any_tasks = False
    for pet in owner.pet_list:
        if pet.tasks:
            any_tasks = True
            st.write(f"**{pet.name}**")
            st.table(
                [
                    {"task": t.description, "category": t.category,
                     "duration": t.duration, "priority": t.priority,
                     "repeat": t.recurrence or "none",
                     "fixed time": t.preferred_time or "—", "status": t.status}
                    for t in view_scheduler.sort_by_priority(pet.tasks)
                ]
            )
    if not any_tasks:
        st.caption("No tasks yet. Add one above.")

st.divider()

# --- Build schedule ----------------------------------------------------------
st.subheader("Build Schedule")

time_available = st.number_input(
    "Time available today (minutes)", min_value=1, max_value=1440, value=90
)

if st.button("Generate schedule"):
    scheduler = Scheduler(owner, int(time_available))
    scheduler.generate_plan()

    conflicts = scheduler.detect_conflicts()
    if conflicts:
        # One prominent, actionable warning. Yellow (st.warning) signals
        # "needs your attention" without the alarm of an error, and we add a
        # concrete next step since a double-booking is something the owner can
        # fix by moving a fixed start time.
        st.warning(
            "⚠️ You're double-booked — you can only be in one place at a "
            "time:\n\n"
            + "\n".join(f"- {warning}" for warning in conflicts)
            + "\n\nMove one task's fixed start time to clear the overlap."
        )

    st.markdown("#### Today's Schedule")
    if scheduler.plan:
        st.table(
            [
                {"start": entry.start_time, "pet": entry.pet.name,
                 "task": entry.task.description,
                 "duration (min)": entry.task.duration,
                 "priority": entry.task.priority}
                for entry in scheduler.plan
            ]
        )
    else:
        st.info("Nothing scheduled — add tasks or increase time available.")

    st.markdown("#### Why this plan")
    # Conflicts are already surfaced above, so drop explain_plan's trailing
    # "Conflicts:" block here to avoid showing the same clash twice.
    explanation = scheduler.explain_plan().split("\nConflicts:")[0]
    st.text(explanation)

st.divider()

# --- Ask PawPal (RAG assistant) ---------------------------------------------
st.subheader("Ask PawPal")

st.caption(
    "Ask a pet-care question and PawPal answers from its trusted care notes, "
    "grounded with sources and aware of your pets. Not a substitute for your vet."
)

question = st.text_input(
    "Your question",
    value="",
    placeholder="e.g. Is chocolate safe for my dog?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("PawPal is checking its notes..."):
            answer = rag.ask_pawpal(
                question.strip(),
                owner=owner,
                knowledge_base=get_knowledge_base(),
            )
        st.markdown(answer.text)

        if answer.citations:
            with st.expander(f"Sources ({len(answer.citations)})"):
                for citation in answer.citations:
                    st.markdown(
                        f"**{citation.source}** — “{citation.quote}”"
                    )
