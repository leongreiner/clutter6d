from pygltflib import GLTF2
g = GLTF2().load("/dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/common/datasets/clutter6d/3d_models/gso_simplified/obj_id_000001__cat_shoe__11pro_SL_TRX_FG.glb")
for i,img in enumerate(g.images or []):
    print(i, "uri=",(img.uri is not None), "bufferView=", img.bufferView, "mime=", img.mimeType)
print("primitives per mesh:", [len(m.primitives) for m in g.meshes or []])