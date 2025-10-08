import streamlit as st
import os

st.title("File System Debugger")

st.header("Current Working Directory")
cwd = os.getcwd()
st.write(f"The current working directory is: `{cwd}`")

st.header("Files in Current Directory")
try:
    files_in_cwd = os.listdir(cwd)
    st.write(files_in_cwd)
    
    # Specifically check for config.yaml
    if 'config.yaml' in files_in_cwd:
        st.success("`config.yaml` was found in the current working directory!")
    else:
        st.error("`config.yaml` was NOT found in the current working directory.")

except Exception as e:
    st.error(f"Could not list files in the current directory: {e}")


st.header("Files in Parent Directory")
try:
    parent_dir = os.path.dirname(cwd)
    st.write(f"The parent directory is: `{parent_dir}`")
    files_in_parent = os.listdir(parent_dir)
    st.write(files_in_parent)
except Exception as e:
    st.error(f"Could not list files in the parent directory: {e}")

st.header("Root Directory Listing")
try:
    root_files = os.listdir('/')
    st.write(root_files)
except Exception as e:
    st.error(f"Could not list files in the root directory: {e}")
    
# Render is known to mount secrets in /etc/secrets
st.header("Checking /etc/secrets (Common location for secrets)")
secrets_dir = "/etc/secrets"
if os.path.exists(secrets_dir):
    try:
        secret_files = os.listdir(secrets_dir)
        st.write(f"Files found in `{secrets_dir}`:")
        st.write(secret_files)
        if 'config.yaml' in secret_files:
            st.success("SUCCESS! `config.yaml` was found in `/etc/secrets`!")
            st.code("/etc/secrets/config.yaml")
    except Exception as e:
        st.error(f"Could not list files in {secrets_dir}: {e}")
else:
    st.warning(f"The directory `{secrets_dir}` does not exist.")