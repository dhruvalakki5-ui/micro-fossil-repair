import streamlit as st
import open3d as o3d
import numpy as np
import tempfile
import os

st.title("Micro-Fossil 3D Surface Repair Tool")
st.markdown("Upload a 3D point cloud file (`.xyz` or `.ply`) to simulate and repair damage!")

uploaded_file = st.file_uploader("Choose a point cloud file", type=["xyz", "ply"])

if uploaded_file is not None:
    # Save uploaded file temporarily so Open3D can read it from disk
    file_extension = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        st.write("Loading point cloud...")
        pcd = o3d.io.read_point_cloud(tmp_path)
        
        st.write(f"Point cloud loaded with {len(pcd.points)} points.")
        
        # Estimate normals with an increased search radius to handle sparse or synthetic sample points
        st.write("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(10)
        
        st.write("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        
        st.success("Reconstruction complete!")
        st.write(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
        
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
