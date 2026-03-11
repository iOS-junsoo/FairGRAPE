import torch
import torch.nn as nn
from torch.autograd import Function
from util import safe_forward_with_cudnn_fallback

############################################################################
# Gradient Reversal Layer (GRL)
############################################################################
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer
    Forward: 입력을 그대로 통과
    Backward: gradient에 -lambda를 곱해서 역전파
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


############################################################################
# Gender Classifier (Adversary)
############################################################################
class GenderClassifier(nn.Module):
    """
    성별 분류기 (MLP 구조)
    입력: 특징 벡터 (feature_dim)
    출력: 성별 로짓 (1차원, BCEWithLogitsLoss 사용)
    """
    def __init__(self, feature_dim, hidden_dim=256):
        super(GenderClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


############################################################################
# Debiased Model Wrapper
############################################################################
class DebiasedModelWrapper(nn.Module):
    """
    기존 모델에 GRL과 Gender Classifier를 추가한 Wrapper
    
    구조:
    - Feature Extractor (Gf): 기존 모델의 features
    - Task Predictor (Gy): 기존 모델의 classifier
    - GRL: Gradient Reversal Layer
    - Gender Classifier (Gd): 성별 예측 adversary
    """
    def __init__(self, base_model, lambda_grl=1.0):
        super(DebiasedModelWrapper, self).__init__()
        
        self.base_model = base_model
        self.lambda_grl = lambda_grl
        
        # 🔥 모델이 있는 디바이스 확인
        device = next(base_model.parameters()).device
        print(f"🔍 Base model device: {device}")
        
        # Feature extractor와 classifier 분리
        if hasattr(base_model, 'features') and hasattr(base_model, 'classifier'):
            # ResNet, MobileNet 등
            self.feature_extractor = base_model.features
            self.task_predictor = base_model.classifier
            
            # Feature dimension 추출
            if hasattr(base_model, 'avgpool'):
                self.avgpool = base_model.avgpool
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Feature dimension 계산
            with torch.no_grad():
                # cuDNN 비활성화 (커스텀 mask conv + BN 안정성)
                prev_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                try:
                    dummy_input = torch.randn(1, 3, 224, 224).to(device)
                    features = self.feature_extractor(dummy_input)
                    if hasattr(self, 'avgpool'):
                        features = self.avgpool(features)
                    features = torch.flatten(features, 1)
                    feature_dim = features.shape[1]
                finally:
                    torch.backends.cudnn.enabled = prev_cudnn
        
        elif hasattr(base_model, 'fc'):
            # Custom model with fc layer
            # 모든 레이어를 feature_extractor로, fc를 task_predictor로
            layers = list(base_model.children())
            self.feature_extractor = nn.Sequential(*layers[:-1])
            self.task_predictor = layers[-1]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Feature dimension
            with torch.no_grad():
                # cuDNN 비활성화 (커스텀 mask conv + BN 안정성)
                prev_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                try:
                    dummy_input = torch.randn(1, 3, 224, 224).to(device)
                    features = self.feature_extractor(dummy_input)
                    features = self.avgpool(features)
                    features = torch.flatten(features, 1)
                    feature_dim = features.shape[1]
                finally:
                    torch.backends.cudnn.enabled = prev_cudnn
        
        else:
            raise ValueError("모델 구조를 분석할 수 없습니다. features/classifier 또는 fc 속성이 필요합니다.")
        
        # GRL 및 Gender Classifier 추가
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        self.gender_classifier = GenderClassifier(feature_dim)
        
        print(f"✅ DebiasedModelWrapper 초기화 완료")
        print(f"   Device: {device}")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Lambda (GRL): {lambda_grl}")
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: 입력 이미지 (B, C, H, W)
            return_features: True면 features도 반환
        
        Returns:
            task_logits: 태스크 예측 (B, num_classes)
            gender_logits: 성별 예측 (B,)
            features (optional): 특징 벡터 (B, feature_dim)
        """
        # Feature extraction
        features = safe_forward_with_cudnn_fallback(self.feature_extractor, x)
        
        # Global average pooling
        if hasattr(self, 'avgpool'):
            features = self.avgpool(features)
        
        # Flatten
        features = torch.flatten(features, 1)
        
        # Task prediction (라벨 예측)
        task_logits = safe_forward_with_cudnn_fallback(self.task_predictor, features)
        
        # Gender prediction through GRL
        reversed_features = self.grl(features)
        gender_logits = safe_forward_with_cudnn_fallback(self.gender_classifier, reversed_features)
        
        if return_features:
            return task_logits, gender_logits, features
        else:
            return task_logits, gender_logits
    
    def get_base_model(self):
        """원래 모델 반환 (학습 후 복원용)"""
        return self.base_model
    
    def set_lambda(self, lambda_grl):
        """GRL의 lambda 값 동적 조정"""
        self.grl.lambda_ = lambda_grl
        self.lambda_grl = lambda_grl


############################################################################
# Pruning Mask 생성 및 적용
############################################################################
def create_pruning_masks(model):
    """
    현재 모델의 가중치가 0인 위치를 마스크로 저장
    
    Returns:
        dict: {parameter_name: mask_tensor}
    """
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype in [torch.float32, torch.float64, torch.float16]:
            # 가중치가 0인 위치는 False, 아닌 위치는 True
            masks[name] = (param.data != 0).float()
    return masks


def apply_pruning_masks(model, masks):
    """
    Pruning mask를 모델에 적용 (가중치가 0이었던 곳은 다시 0으로)
    
    Args:
        model: 모델
        masks: create_pruning_masks로 생성한 마스크 dict
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]


def apply_pruning_masks_to_gradients(model, masks):
    """
    Gradient에 mask 적용 (가중치가 0인 위치의 gradient를 0으로)
    optimizer.step() 이전에 호출
    
    Args:
        model: 모델
        masks: create_pruning_masks로 생성한 마스크 dict
    """
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad *= masks[name]