import cv2
import numpy as np
import json
import os
from datetime import datetime
from loma import LoMa, LoMaB128, LoMaB, LoMaL, LoMaG, LoMaR


MODEL_CONFIGS = {
    "loma_b128": LoMaB128,
    "loma_b": LoMaB,
    "loma_l": LoMaL,
    "loma_g": LoMaG,
    "loma_r": LoMaR,
}


class LoMaMatcher:
    def __init__(self, model=None, model_name="loma_b128", save_dir="./loma_results"):
        if model is None:
            if model_name not in MODEL_CONFIGS:
                available = ", ".join(MODEL_CONFIGS)
                raise ValueError(f"未知模型: {model_name}. 可选: {available}")
            print(f"[INFO] 正在加载模型: {model_name}")
            self.model = LoMa(MODEL_CONFIGS[model_name]())
        else:
            self.model = model
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 为本次运行创建独立子目录
        self.run_dir = os.path.join(
            self.save_dir, 
            datetime.now().strftime("%m%d_%H%M%S")
        )
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"[INFO] 结果将保存至: {self.run_dir}")

    def match_and_estimate(self, img_path_A, img_path_B, 
                           ransac_thresh=0.5, 
                           confidence=0.999999,
                           max_iters=10000):
        """
        执行LoMa匹配 + 基础矩阵估计
        """
        # 1. LoMa 特征匹配
        print(f"[INFO] 正在匹配: {img_path_A} <-> {img_path_B}")
        kptsA, kptsB = self.model.match(img_path_A, img_path_B)
        
        if len(kptsA) == 0:
            print("[WARN] 未找到任何匹配点!")
            return None
        
        # 2. 基础矩阵估计 (RANSAC)
        F, mask = cv2.findFundamentalMat(
            kptsA,
            kptsB,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=confidence,
            maxIters=max_iters,
        )
        
        # 3. 统计信息
        inliers = int(mask.sum()) if mask is not None else 0
        total = len(kptsA)
        ratio = inliers / max(total, 1)
        
        stats = {
            "total_matches": total,
            "inliers": inliers,
            "outliers": total - inliers,
            "inlier_ratio": round(ratio, 4),
            "fundamental_matrix": F.tolist() if F is not None else None,
            "ransac_threshold": ransac_thresh,
            "confidence": confidence,
        }
        
        print(f"[RESULT] 总匹配: {total}")
        print(f"[RESULT] 内点: {inliers} | 外点: {total - inliers}")
        print(f"[RESULT] 内点率: {ratio:.4f}")
        
        # 4. 保存统计数据
        stats_path = os.path.join(self.run_dir, "match_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] 统计信息已保存: {stats_path}")
        
        return {
            "kptsA": kptsA,
            "kptsB": kptsB,
            "F": F,
            "mask": mask,
            "stats": stats,
            "img_path_A": img_path_A,
            "img_path_B": img_path_B,
        }

    def visualize(self, result, max_draw=300, save_name="matches_visualization.jpg"):
        """
        可视化匹配结果：内点(绿色) vs 外点(红色)
        """
        if result is None:
            print("[WARN] 无结果可可视化")
            return
        
        kptsA = result["kptsA"]
        kptsB = result["kptsB"]
        mask = result["mask"]
        imgA = cv2.imread(result["img_path_A"])
        imgB = cv2.imread(result["img_path_B"])
        
        if imgA is None or imgB is None:
            raise ValueError("无法读取图像，请检查路径")
        
        # 转换为 KeyPoint 对象
        kpA = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kptsA]
        kpB = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kptsB]
        
        # 构建 DMatch 列表
        matches = []
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(kpA), dtype=bool)
        
        for i in range(len(kpA)):
            matches.append(cv2.DMatch(i, i, 0))
        
        # 限制绘制数量避免过于拥挤
        if len(matches) > max_draw:
            # 优先保留内点，随机采样外点
            inlier_indices = np.where(inlier_mask)[0]
            outlier_indices = np.where(~inlier_mask)[0]

            rng = np.random.default_rng(42)

            if len(inlier_indices) >= max_draw:
                keep_indices = rng.choice(inlier_indices, max_draw, replace=False)
            else:
                remaining = max_draw - len(inlier_indices)
                if len(outlier_indices) > remaining:
                    sampled_outliers = rng.choice(
                        outlier_indices,
                        remaining,
                        replace=False
                    )
                else:
                    sampled_outliers = outlier_indices
                keep_indices = np.concatenate([inlier_indices, sampled_outliers])

            keep_indices = np.sort(keep_indices)
            kpA = [kpA[i] for i in keep_indices]
            kpB = [kpB[i] for i in keep_indices]
            inlier_mask = inlier_mask[keep_indices]

            # 裁剪关键点后必须重建 DMatch，避免 queryIdx/trainIdx 仍指向原始索引。
            matches = [cv2.DMatch(i, i, 0) for i in range(len(kpA))]
        
        # 绘制参数
        # 内点: 绿色线条, 外点: 红色线条
        draw_params = dict(
            matchColor=(0, 0, 255),      # 默认红色（外点）
            singlePointColor=None,
            matchesMask=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 手动绘制以区分内外点颜色
        hA, wA = imgA.shape[:2]
        hB, wB = imgB.shape[:2]
        h_max = max(hA, hB)
        vis = np.zeros((h_max, wA + wB, 3), dtype=np.uint8)
        vis[:hA, :wA] = imgA
        vis[:hB, wA:wA+wB] = imgB
        
        # 绘制所有匹配线
        for i, m in enumerate(matches):
            ptA = (int(kpA[m.queryIdx].pt[0]), int(kpA[m.queryIdx].pt[1]))
            ptB = (int(kpB[m.trainIdx].pt[0] + wA), int(kpB[m.trainIdx].pt[1]))
            
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)  # 绿=内点, 红=外点
            cv2.line(vis, ptA, ptB, color, 1, cv2.LINE_AA)
            cv2.circle(vis, ptA, 3, color, -1, cv2.LINE_AA)
            cv2.circle(vis, ptB, 3, color, -1, cv2.LINE_AA)
        
        # 添加图例和统计信息
        legend_y = 30
        cv2.putText(vis, f"Inliers (Green): {result['stats']['inliers']}", 
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Outliers (Red): {result['stats']['outliers']}", 
                    (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Ratio: {result['stats']['inlier_ratio']:.2%}", 
                    (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存可视化结果
        save_path = os.path.join(self.run_dir, save_name)
        cv2.imwrite(save_path, vis)
        print(f"[SAVE] 可视化结果已保存: {save_path}")
        
        return vis

    def export_matches(self, result, save_name="matches_data.npz"):
        """
        导出原始匹配数据为 npz 格式，便于后续分析
        """
        if result is None:
            return
        
        kptsA = result["kptsA"]
        kptsB = result["kptsB"]
        mask = result["mask"].ravel() if result["mask"] is not None else np.ones(len(kptsA))
        F = result["F"]
        
        save_path = os.path.join(self.run_dir, save_name)
        np.savez(
            save_path,
            kptsA=kptsA,
            kptsB=kptsB,
            inlier_mask=mask.astype(bool),
            fundamental_matrix=F,
            imgA_path=result["img_path_A"],
            imgB_path=result["img_path_B"],
        )
        print(f"[SAVE] 匹配数据已导出: {save_path}")
        return save_path



def main():
    # ==================== 配置 ====================
    IMG_A = "assets/c.jpeg"   # 替换为你的图片路径
    IMG_B = "assets/d.jpeg"   # 替换为你的图片路径
    SAVE_DIR = "./results"
    #MODEL_NAME = "loma_b128"  # 可选: loma_b128, loma_b, loma_l, loma_g, loma_r
    MODEL_NAME = "loma_b"
    # MODEL_NAME = "loma_l"
    # MODEL_NAME = "loma_g"
    # MODEL_NAME = "loma_r"

    # ==================== 执行 ====================
    matcher = LoMaMatcher(model_name=MODEL_NAME, save_dir=SAVE_DIR)
    
    # 1. 匹配并估计基础矩阵
    result = matcher.match_and_estimate(
        IMG_A, IMG_B,
        ransac_thresh=0.5,
        confidence=0.999999,
        max_iters=10000
    )
    
    if result:
        # 2. 可视化并保存
        vis = matcher.visualize(result, max_draw=300)
        
        # 3. 导出原始数据
        # matcher.export_matches(result)

        # 4. 可选：实时显示（如果环境支持GUI）
        # cv2.imshow("LoMa Matches", vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
