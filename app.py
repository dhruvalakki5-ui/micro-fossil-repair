import numpy as np
import open3d as o3d
import streamlit as st

st.title("Micro-Fossil 3D Surface Repair Tool")
st.write(
    "Upload a 3D point cloud file (.xyz or .ply) to simulate and repair"
    " damage!"
)

uploaded_file = st.file_uploader(
    "Choose a point cloud file", type=["xyz", "ply"]
)

if uploaded_file is not None:
  pcd = o3d.io.read_point_cloud(uploaded_file.name)
  st.write("Processing point cloud and running Poisson reconstruction...")

  pcd.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
  )
  mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
      pcd, depth=8
  )

  output_filename = "repaired_model.ply"
  o3d.io.write_triangle_mesh(output_filename, mesh)

  with open(output_filename, "rb") as file:
    st.download_button(
        label="Download Repaired 3D Mesh",
        data=file,
        file_name="repaired_model.ply",
        mime="application/octet-stream",
    )
  st.success("Repair complete!")