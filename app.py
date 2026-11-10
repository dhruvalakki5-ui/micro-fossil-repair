import streamlit as st
import open3d as o3d
import numpy as np
import tempfile
import os
from streamlit_stl import st_stl

st.title("Micro-Fossil 3D Surface Repair Tool")
st.markdown("Upload a 3D point cloud file (`.xyz` or `.ply`) to simulate and repair damage!")

uploaded_file = st.file_uploader("Choose a point cloud file", type=["xyz", "ply"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        st.write("Loading point cloud...")
        pcd = o3d.io.read_point_cloud(tmp_path)
        
        st.write(f"Point cloud loaded with {len(pcd.points)} points.")
        
        st.write("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(10)
        
        st.write("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        
        st.success("Reconstruction complete!")
        st.write(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
        
        # Save mesh as STL for browser rendering
        stl_path = tempfile.NamedTemporaryFile(delete=False, suffix=".stl").name
        o3d.io.write_triangle_mesh(stl_path, mesh)
        
        st.subheader("Interactive 3D Mesh Viewer")
        st_stl(stl_path, height=400)
        
        # Download button
        with open(stl_path, "rb") as f:
            st.download_button(
                label="📥 Download Repaired 3D Mesh (.stl)",
                data=f,
                file_name="repaired_fossil_mesh.stl",
                mime="application/octet-stream"
            )
            
        if os.path.exists(stl_path):
            os.remove(stl_path)
        
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
