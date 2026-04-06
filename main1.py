import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import pdfplumber


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)
print("API key loaded")


chroma_client = PersistentClient(path="./chroma_db")


SYSTEM_PROMPT = """
You are an expert construction project assistant.

You help construction workers understand project documents.

Rules:
- Use the provided project documents to answer
- Be clear and practical
- If information is not in the documents, say you are unsure
- Never give legal guarantees
"""

def get_projects():
    projects_folder = "projects"
    return [
        p for p in os.listdir(projects_folder)
        if os.path.isdir(os.path.join(projects_folder, p))
    ]


def load_project_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""

                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "\n".join(
                            [" | ".join([str(cell) for cell in row]) for row in table]
                        )
                        text += "\n" + table_text

                    if text.strip():
                        documents.append(text)

        elif file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                documents.append(f.read())

    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_project_index(project_path, project_name):
    project_docs = load_project_documents(project_path)

    chunks = []
    for doc in project_docs:
        chunks.extend(chunk_text(doc))

    print(f"{len(chunks)} document chunks created for {project_name}")

    print("Creating embeddings (batched)...")


    batch_size = 50
    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        embeddings.extend([item.embedding for item in response.data])

    collection = chroma_client.get_or_create_collection(name="construction_docs")

    collection.delete(where={"project": project_name})

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{project_name}_{i}" for i in range(len(chunks))],
        metadatas=[{"project": project_name} for _ in chunks]
    )
    

    print(f"ChromaDB collection built for project: {project_name}\n")


    return collection

def select_project(projects):
    print("\nAvailable Projects:")
    for i, project in enumerate(projects):
        print(f"{i+1}. {project}")

    while True:
        user_choice = input("\nSelect project (name or number): ").strip().lower()

        if user_choice.isdigit():
            index = int(user_choice) - 1
            if 0 <= index < len(projects):
                return projects[index]
            else:
                print("Invalid number. Try again.")

        else:
            matches = [p for p in projects if user_choice in p.lower()]

            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                print("Multiple matches found. Be more specific.")
            else:
                print("Project not found. Try again.")

def main():
    projects = get_projects()

    if len(projects) == 0:
        raise ValueError("No projects found in /projects folder")

    selected_project = select_project(projects)
    project_path = f"projects/{selected_project}"

    collection = build_project_index(project_path, selected_project)

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurrently active project: {selected_project}"}
    ]

    print("\nConstruction Assistant Ready")
    print("Type 'quit' to exit")
    print("Type 'switch project' or 'change project' to switch projects anytime\n")

    while True:
        user_input = input("Worker: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() in ["switch project", "change project"]:
            selected_project = select_project(projects)
            project_path = f"projects/{selected_project}"
            collection = build_project_index(project_path, selected_project)

            conversation[0]["content"] = f"{SYSTEM_PROMPT}\nCurrently active project: {selected_project}"

            print(f"\nSwitched to project: {selected_project}\n")
            continue

        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_input
        ).data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"project": selected_project}
        )

        relevant_chunks = "\n\n".join(results["documents"][0])

        conversation.append({
            "role": "user",
            "content": f"""
Worker Question:
{user_input}

Relevant Project Documents:
{relevant_chunks}
"""
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation
        )

        reply = response.choices[0].message.content

        print("\nAssistant:", reply, "\n")

        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()