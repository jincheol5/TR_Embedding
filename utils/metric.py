import torch
from sklearn.metrics import average_precision_score,roc_auc_score

class Metric:
    @staticmethod
    def compute_ACC(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Input:
            pred_logit
            label
        Return:
            acc
        """
        pred=(torch.sigmoid(pred_logit)>=0.5).float()
        acc=(pred==label.float()).float().mean().item()
        return acc

    @staticmethod
    def compute_AP(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        AP (Average Precision)
        -> 점수 상위에 실제 연결이 얼마나 잘 모여 있는지 평가.
        값 범위: 0 ~ 1
            1.0: 실제 연결들이 모든 가짜 연결보다 높은 점수를 받음.
            높을수록: 상위 예측에서 실제 연결을 더 정확하게 찾아냄.
            무작위 예측의 기준 수준: 평가 데이터의 양성 비율. 현재처럼 양성·음성이 1:1이면 약 0.5.
        """
        pred_score=torch.sigmoid(pred_logit.detach()).cpu().reshape(-1).numpy()
        true_label=label.detach().cpu().reshape(-1).numpy()
        return float(average_precision_score(true_label,pred_score))

    @staticmethod
    def compute_AUC(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        ROC-AUC
        -> 실제 연결에 가짜 연결보다 높은 점수를 주는 정도 측정.
        값 범위: 0 ~ 1
            1.0: 실제 연결과 가짜 연결의 순위를 완벽하게 구분.
            0.5: 무작위 수준의 구분 능력.
            0.5 미만: 실제 연결에 오히려 더 낮은 점수를 주는 경향.
            0.0: 모든 실제 연결이 모든 가짜 연결보다 낮은 점수를 받음.
        """
        pred_score=torch.sigmoid(pred_logit.detach()).cpu().reshape(-1).numpy()
        true_label=label.detach().cpu().reshape(-1).numpy()
        return float(roc_auc_score(true_label,pred_score))