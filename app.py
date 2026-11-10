import streamlit as st
import open3d as o3d
import numpy as np
import tempfile
import os
import plotly.graph_objects as go

st.title("Micro-Fossil 3D Surface Repair Tool")
st.markdown("Upload a 3D point cloud file (`.xyz` or `.ply`) to simulate and repair damage!")

# Sidebar color picker for users to choose their own mesh color
st.sidebar.header("Viewer Customization")
user_mesh_color = st.sidebar.color_picker("Choose Mesh Color", "#d4a373")

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
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if len(vertices) > 0 and len(triangles) > 0:
            st.subheader("Interactive 3D Mesh Viewer")
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    color=user_mesh_color,    # Uses the color picked by the user!
                    opacity=0.9,
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.8,
                        specular=0.5,
                        roughness=0.5
                    ),
                    lightposition=dict(
                        x=100,
                        y=200,
                        z=300
                    )
                )
            ])
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        output_mesh_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ply").name
        o3d.io.write_triangle_mesh(output_mesh_path, mesh)
        
        with open(output_mesh_path, "rb") as f:
            st.download_button(
                label="📥 Download Repaired 3D Mesh (.ply)",
                data=f,
                file_name="repaired_fossil_mesh.ply",
                mime="application/octet-stream"
            )
            
        if os.path.exists(output_mesh_path):
            os.remove(output_mesh_path)
        
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
