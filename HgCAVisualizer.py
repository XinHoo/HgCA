import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from data_util import ModelNet40

class HgCAVisualizer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        from models.HgCA import Backbone
        from models.ResMLP import MLPBlockFC

        self.backbone = Backbone(cfg)
        self.mlp1 = MLPBlockFC(cfg.patch_dim[-1], 512, cfg.dropout)
        self.mlp2 = MLPBlockFC(512, 256, cfg.dropout)
        self.output_layer = nn.Linear(256, cfg.num_classes)

    def forward(self, x):
        patches, pos_and_feats,pos_and_avgfeats,hyperedges = self.backbone(x)
        res = torch.max(patches, dim=1)[0]       
        res = self.mlp2(self.mlp1(res))
        res = self.output_layer(res)             
        return res, pos_and_feats, pos_and_avgfeats,hyperedges

def save_point_cloud_visualization(points, colors=None,
                                   title="Point Cloud",
                                   save_path="visualization.png",
                                   stage_name=""):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=colors, cmap='viridis', s=200, alpha=1.0)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c='blue', s=200, alpha=1.0)
    ax.grid(False)
    ax.set_axis_off()

    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")

def euler_deg_to_matrix(yaw, pitch, roll, order='zyx'):

    y = math.radians(yaw)
    p = math.radians(pitch)
    r = math.radians(roll)

    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(p), math.sin(p)
    cx, sx = math.cos(r), math.sin(r)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)

    mapping = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3, dtype=np.float32)
    for ax in order.lower():
        R = mapping[ax] @ R
    return R


def apply_rotation(points_xyz, R):

    return (R @ points_xyz.T).T


def visualize_hgca_stages(model, data_loader, output_dir="visualizations",
                               num_samples=5, device=None,
                               rot_global=None, rot_order='zyx', rot_table=None):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    class_names = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
        'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
        'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
        'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]

    sample_count = 0
    rot_table = rot_table or {}

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            if sample_count >= num_samples:
                break


            data = data.to(device, non_blocking=True).float()
            labels = labels.to(device)


            bs = data.shape[0]
            for i in range(min(bs, num_samples - sample_count)):
                single_data = data[i:i + 1]    
                single_label = int(labels[i].item())

                try:
                    predictions, pos_and_feats, pos_and_avgfeats, hyperedges = model(single_data)
                    pred_class = int(torch.argmax(predictions, dim=1).item())

                    sample_idx_1based = sample_count + 1
                    sample_name = f"sample_{sample_idx_1based}_{class_names[single_label]}"
                    print(f"\n[Info] Processing {sample_name}")
                    print(f"[Info] GT: {class_names[single_label]}, Pred: {class_names[pred_class]}")


                    for stage_idx, stage in enumerate(hyperedges):
                        pos, cluster_ids = stage 

                        stage_name = f"Stage_{stage_idx}"
                        pos_np = pos[0].detach().cpu().numpy().astype(np.float32)
                        cluster_ids_np = cluster_ids[0].detach().cpu().numpy().astype(np.int32)


                        if rot_global is not None:
                            pos_np = apply_rotation(pos_np, rot_global)
                        if sample_idx_1based in rot_table:
                            y2, p2, r2 = rot_table[sample_idx_1based]
                            R2 = euler_deg_to_matrix(y2, p2, r2, order=rot_order)
                            pos_np = apply_rotation(pos_np, R2)


                        if cluster_ids_np.ndim > 1:
                            cluster_ids_single = cluster_ids_np[0]   
                        else:
                            cluster_ids_single = cluster_ids_np

                        cluster_ids_single = np.asarray(cluster_ids_single).flatten()  
                        num_points = pos_np.shape[0]

                        if cluster_ids_single.size != num_points:

                            if cluster_ids_single.size < num_points:
                                reps = (num_points + cluster_ids_single.size - 1) // cluster_ids_single.size
                                cluster_ids_single = np.tile(cluster_ids_single, reps)[:num_points]
                            else:
                                cluster_ids_single = cluster_ids_single[:num_points]

                        num_clusters = cluster_ids_single.max() + 1
                        cmap = plt.get_cmap("tab20")
                        colors = cmap(cluster_ids_single % 20)[:, :3]  

                        save_path = os.path.join(output_dir, f"{sample_name}_{stage_name}.png")
                        save_point_cloud_visualization(
                            pos_np, colors,
                            title=f"{class_names[single_label]} (Pred: {class_names[pred_class]})",
                            save_path=save_path,
                            stage_name=f"{stage_name} - {pos_np.shape[0]} points"
                        )

                    sample_count += 1

                except Exception as e:
                    print(f"[Error] forward failed on sample #{sample_count + 1}: {e}")
                    continue

                if sample_count >= num_samples:
                    break

    print(f"\n[Done] Generated {sample_count} sample visualizations at '{output_dir}'")


def load_model_config():

    config = {
        'model_name': 'HgCA',
        'dataset': 'ModelNet40',
        'dataset_dir': 'data/',
        'num_classes': 40,
        'num_points': 1024,
        'num_heads': 4,
        'down_ratio': [1, 2, 4, 8, 16],
        'patch_size': [16, 16, 16, 16, 16],
        'local_size': [16, 16, 16, 16, 16],
        'patch_dim': [3, 64, 128, 256, 512, 1024],  
        'gpu': 0,
        'test_batch_size': 1,
        'seed': 9344
    }
    return OmegaConf.create(config)

def main():
    parser = argparse.ArgumentParser(description='HgCA Stage Visualization (with rotation)')
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='Path to trained model (ckpt with or without model_state_dict)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=15,
                        help='Number of samples to visualize')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Dataset root directory')
    parser.add_argument('--only_table', action='store_true', default=True,
                        help='Only visualize the "table" class (index 33)')
    parser.add_argument('--rot_euler', type=str, default="0,180,90",
                        help='"yaw,pitch,roll"')
    parser.add_argument('--rot_order', type=str, default='zyx',
                        help='zyx')
    parser.add_argument('--rot_per_sample_csv', type=str, default=None,
                        help='idx,yaw,pitch,roll')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    cfg = load_model_config()
    cfg.dataset_dir = args.data_dir


    print("[Info] Loading dataset...")
    try:
        test_dataset = ModelNet40(cfg.dataset_dir, partition='test', num_points=cfg.num_points)
        print(f"[Info] Test samples: {len(test_dataset)}")

        if args.only_table:
            table_samples = []
            for i in range(len(test_dataset)):
                data_i, label_i = test_dataset[i]
                if int(label_i) == 33:  # chair 8 table 33 airplane 0
                    # numpy -> tensor
                    if isinstance(data_i, np.ndarray):
                        data_i = torch.from_numpy(data_i).float()
                    else:
                        data_i = data_i.float() if torch.is_tensor(data_i) else torch.tensor(data_i, dtype=torch.float32)
                    table_samples.append((data_i, int(label_i)))
                if len(table_samples) >= args.num_samples:
                    break

            if len(table_samples) > 0:
                print(f"[Info] Found {len(table_samples)} 'airplane' samples for visualization.")
                table_data = torch.stack([s[0] for s in table_samples])                      # (K, N, 3)
                table_labels = torch.tensor([s[1] for s in table_samples], dtype=torch.long) # (K,)
                table_dataset = TensorDataset(table_data, table_labels)
                test_loader = DataLoader(table_dataset, batch_size=1, shuffle=False, drop_last=False)
            else:
                print("[Warn] No 'table' samples found, fallback to full test set.")
                test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, drop_last=False)
        else:
            test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, drop_last=False)

    except Exception as e:
        print(f"[Warn] Dataset load failed: {e}")
        print("[Info] Creating dummy data for demonstration...")
        dummy_data = torch.randn(args.num_samples, cfg.num_points, 3)
        dummy_labels = torch.randint(0, cfg.num_classes, (args.num_samples,), dtype=torch.long)
        dummy_dataset = TensorDataset(dummy_data, dummy_labels)
        test_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, drop_last=False)

    rot_table = {}
    if args.rot_per_sample_csv is not None and os.path.exists(args.rot_per_sample_csv):
        import csv
        with open(args.rot_per_sample_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:

                k = int(row['idx'])
                yaw = float(row.get('yaw', 0.0))
                pitch = float(row.get('pitch', 0.0))
                roll = float(row.get('roll', 0.0))
                rot_table[k] = (yaw, pitch, roll)
        print(f"[Info] Loaded per-sample rotations: {len(rot_table)} entries.")


    rot_global = None
    if args.rot_euler is not None:
        try:
            yaw, pitch, roll = [float(x) for x in args.rot_euler.split(',')]
            rot_global = euler_deg_to_matrix(yaw, pitch, roll, order=args.rot_order)
            print(f"[Info] Global rotation (deg): yaw={yaw}, pitch={pitch}, roll={roll}, order={args.rot_order}")
        except Exception as e:
            print(f"[Warn] Failed to parse --rot_euler: {args.rot_euler}. {e}")

    print("[Info] Initializing model...")
    model = HgCAVisualizer(cfg).to(device)

    if os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            state = checkpoint.get('model_state_dict', checkpoint) 
            model.load_state_dict(state, strict=False)
            print("[Info] Model loaded successfully!")
        except Exception as e:
            print(f"[Warn] Load checkpoint failed: {e}\n[Info] Use randomly initialized model.")
    else:
        print(f"[Warn] Checkpoint not found: {args.model_path}. Use randomly initialized model.")

    print("[Info] Start visualization...")
    visualize_hgca_stages(
        model, test_loader, args.output_dir, args.num_samples, device=device,
        rot_global=rot_global, rot_order=args.rot_order, rot_table=rot_table
    )


if __name__ == "__main__":
    main()
