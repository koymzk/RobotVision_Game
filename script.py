import cv2
import pygame
import mediapipe as mp
import os
import math
import numpy as np

# サウンド管理クラス
class SoundManager:
    _initialized = False
    _base_dir = "./sounds"
    _sounds = {}

    @classmethod
    def init(cls, base_dir="./sounds"):
        if not cls._initialized:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"[SoundManager] mixer init failed: {e}")
            cls._base_dir = base_dir
            cls._initialized = True

    @classmethod
    def load(cls, key, filename, volume=1.0):
        if not cls._initialized:
            cls.init()
        path = os.path.join(cls._base_dir, filename)
        try:
            snd = pygame.mixer.Sound(path)
            snd.set_volume(volume)
            cls._sounds[key] = snd
        except Exception as e:
            print(f"[SoundManager] load failed for '{path}': {e}")

    @classmethod
    def play(cls, key):
        snd = cls._sounds.get(key)
        if snd is not None:
            try:
                snd.play()
            except Exception as e:
                print(f"[SoundManager] play failed for '{key}': {e}")
        else:
            print(f"[SoundManager] sound not loaded: {key}")

class Player:
    def __init__(self, health, attack_power, name="P"):
        self.health = health
        self.max_health = health
        self.attack_power = attack_power
        self.damage_flash = False  # ダメージフラッシュの状態
        self.flash_duration = 5  # フラッシュの持続時間
        self.flash_timer = 0
        # --- Pose / Action 関連 ---
        self.pose_landmarks = None  # 最新の姿勢ランドマーク(ワールド座標ではなく画像座標)
        self.last_action = None     # 直近に検出したアクション名（"punch" など）
        # 識別用
        self.name = name
        self.opponent = None
        # ガード検出器（プレイヤーごとに独立）
        self.guard_detector = GuardDetector(player_name=self.name)
        # パンチ検出器（プレイヤーごとに独立）
        self.punch_detector = PunchDetector(player_name=self.name, owner=self)
        self.hit_event = None  # {'shoulder': 'left'|'right', 't0': ms}
        self.guard_block_event = {'left': None, 'right': None}  # ガード成功の発光トリガ（手ごと）
        # ガード反射ダメージ用のフラッシュ（枠表示）
        self.reflect_event = None  # {'t0': ms, 'dur': ms}
        self.laser_event = None
    
    def take_damage(self, hand=None, amount=None):
        SoundManager.play("beam")
        dmg = self.attack_power if amount is None else amount
        self.health -= dmg
        try:
            t0 = pygame.time.get_ticks()
        except Exception:
            t0 = 0
        # 攻撃手の指定がない場合は被弾エフェクトなし
        shoulder = 'left' if hand == 'left' else ('right' if hand == 'right' else None)
        self.hit_event = None if shoulder is None else {
            'shoulder': shoulder,
            't0': t0,
        }

    def take_reflect_damage(self, amount):
        # 反射ダメージ：サウンドなし、肩エフェクトなし
        dmg = self.attack_power if amount is None else amount
        self.health -= dmg
        try:
            t0 = pygame.time.get_ticks()
        except Exception:
            t0 = 0
        # 反射専用の画面枠フラッシュ（0.5秒）
        self.reflect_event = {'t0': t0, 'dur': 500}
    
    def update_flash(self):
        if self.damage_flash:
            self.flash_timer -= 1
            if self.flash_timer <= 0:
                self.damage_flash = False

    # Pose関連: Mediapipeからの結果を保持
    def set_pose_landmarks(self, landmarks):
        self.pose_landmarks = landmarks

    # 将来的なアクション判定用のフック。ActionRecognizerに委譲
    def update_action(self, recognizer):
        if recognizer is None:
            return
        self.last_action = recognizer.recognize(self.pose_landmarks)
        # ここで last_action に応じてメソッド呼び出しなどを行える
        # 例: if self.last_action == "punch": self.punch()

    def update_guard(self, dt_ms):
        # Mediapipeのランドマークがあるときのみ更新
        if self.pose_landmarks is not None:
            self.guard_detector.update(self.pose_landmarks, dt_ms)

    def update_punch(self, dt_ms):
        if self.pose_landmarks is not None:
            self.punch_detector.update(self.pose_landmarks, dt_ms)

    def draw_health_bar(self, screen, frame_width, frame_height):
        health_bar_width = frame_width * 0.4
        health_bar_height = 20
        health_bar_x = (frame_width - health_bar_width) // 2  # 中央に配置
        health_bar = pygame.Rect(health_bar_x, 20, health_bar_width, health_bar_height)
        pygame.draw.rect(screen, (255, 0, 0), health_bar)  # 赤い枠
        ratio = 0 if self.max_health <= 0 else max(0.0, min(1.0, self.health / self.max_health))
        pygame.draw.rect(screen, (0, 255, 0), (health_bar_x, 20, ratio * health_bar_width, health_bar_height))  # 緑の体力ゲージ
        # --- HP数値（バー直下中央、フォントはバー高さの約2倍） ---
        try:
            hp_font_size = max(12, int(health_bar_height * 2))
            hp_font = pygame.font.SysFont(None, hp_font_size)
            hp_text = str(int(max(0, self.health)))  # 現在HPのみ表示
            hp_surf = hp_font.render(hp_text, True, (255, 255, 255))
            hp_rect = hp_surf.get_rect()
            hp_rect.midtop = (int(health_bar_x + health_bar_width / 2), int(20 + health_bar_height + 2))
            screen.blit(hp_surf, hp_rect)
        except Exception:
            pass
        # --- Guard Stamina Circles (HPゲージ左右) ---
        def _draw_pie(surface, center, radius, fraction, color):
            # fraction in [0,1]
            if fraction <= 0:
                return
            cx, cy = center
            steps = max(2, int(60 * fraction))
            theta_max = 2 * math.pi * fraction
            pts = [(cx, cy)]
            for i in range(steps + 1):
                t = (i / steps) * theta_max - math.pi / 2  # 上(12時)から時計回り
                x = int(cx + radius * math.cos(t))
                y = int(cy + radius * math.sin(t))
                pts.append((x, y))
            try:
                pygame.draw.polygon(surface, color, pts)
            except Exception:
                pass

        gd = getattr(self, 'guard_detector', None)
        if gd is not None:
            radius = int(health_bar_height * 3)  # 1.5倍（従来: 2*h → 3*h）
            cy = 20 + health_bar_height // 2
            pad = 12
            # 左右を逆に配置し、下方向へ半径分オフセット
            left_center = (int(health_bar_x + health_bar_width + pad + radius), int(cy + radius))
            right_center = (int(health_bar_x - pad - radius), int(cy + radius))
            # 背景（欠け）灰色の円
            pygame.draw.circle(screen, (80, 80, 80), left_center, radius)
            pygame.draw.circle(screen, (80, 80, 80), right_center, radius)
            # 左手
            lsmax = max(1, int(getattr(gd, 'left_stamina_max_ms', 3000)))
            lsval = max(0, int(getattr(gd, 'left_stamina_ms', 0)))
            lsr = max(0.0, min(1.0, lsval / lsmax))
            l_charging = getattr(gd, 'left_lockout', False) and (lsval < lsmax)
            l_color = (255, 255, 0) if l_charging else (0, 0, 255)
            _draw_pie(screen, left_center, radius, lsr, l_color)
            # 右手
            rsmax = max(1, int(getattr(gd, 'right_stamina_max_ms', 3000)))
            rsval = max(0, int(getattr(gd, 'right_stamina_ms', 0)))
            rsr = max(0.0, min(1.0, rsval / rsmax))
            r_charging = getattr(gd, 'right_lockout', False) and (rsval < rsmax)
            r_color = (255, 255, 0) if r_charging else (0, 0, 255)
            _draw_pie(screen, right_center, radius, rsr, r_color)
            # 枠線
            pygame.draw.circle(screen, (200, 200, 200), left_center, radius, 1)
            pygame.draw.circle(screen, (200, 200, 200), right_center, radius, 1)

# Mediapipe Pose をラップする検出クラス（1カメラ=1インスタンス推奨）
class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, draw_landmarks=False):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.draw_landmarks = draw_landmarks

    def process(self, frame_rgb):
        """
        入力: RGBの画像(numpy array)
        出力: (人物のみRGBA画像, landmarks)
        """
        results = self.pose.process(frame_rgb)
        landmarks = results.pose_landmarks

        # --- セグメンテーションを用いて人物のみRGBAで返す（人物=不透明/背景=透明） ---
        seg_mask = getattr(results, "segmentation_mask", None)

        # ランドマーク描画はまずRGBに行う
        output_rgb = frame_rgb.copy()
        if landmarks is not None and self.draw_landmarks:
            self.mp_drawing.draw_landmarks(
                output_rgb,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )

        h, w = output_rgb.shape[:2]
        if seg_mask is not None:
            # エッジを少しシャープにするソフトしきい値
            alpha_f = np.clip((seg_mask - 0.1) / 0.2, 0.0, 1.0)  # [0,1]
            alpha = (alpha_f * 255).astype(np.uint8)
        else:
            # セグメントが無い場合はカメラを隠す（背景だけ見える）
            alpha = np.zeros((h, w), dtype=np.uint8)

        output_rgba = np.dstack([output_rgb, alpha])  # RGBA
        return output_rgba, landmarks

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None

class GuardDetector:
    """
    肘→手首ベクトルがほぼ垂直上（画像座標では上=Yが小さい）かどうかを判定し、
    その状態が0.5秒継続でガードON、外れた状態が0.5秒継続でガードOFF。
    左右独立、プレイヤー毎に独立。
    """
    def __init__(self, player_name="P", angle_ratio_thresh=1.0, hold_ms=500):
        self.player_name = player_name
        self.angle_ratio_thresh = angle_ratio_thresh  # |dx| <= thresh * |dy| を「ほぼ垂直」とする
        self.hold_ms = hold_ms
        # 状態
        self.left_active = False
        self.right_active = False
        # 継続時間の蓄積
        self.left_in_ms = 0
        self.left_out_ms = 0
        self.right_in_ms = 0
        self.right_out_ms = 0
        # --- Guard Stamina（左右独立・各3秒） ---
        self.left_stamina_max_ms = 3000
        self.left_stamina_ms = self.left_stamina_max_ms
        self.left_lockout = False
        self.right_stamina_max_ms = 3000
        self.right_stamina_ms = self.right_stamina_max_ms
        self.right_lockout = False

    def _is_vertical_up(self, wrist, elbow):
        # Mediapipeのlandmarkは正規化座標 [0,1]。画像座標系でyは下に向かって増加
        dx = wrist.x - elbow.x
        dy = wrist.y - elbow.y
        # 上向き条件: 手首が肘より上（yが小さい）
        up = dy < 0
        # 垂直度: 水平成分が小さい
        vertical_enough = abs(dx) <= self.angle_ratio_thresh * abs(dy) if dy != 0 else False
        return up and vertical_enough

    def _get_lm(self, landmarks, idx):
        # landmarks は results.pose_landmarks
        return landmarks.landmark[idx]

    def update(self, landmarks, dt_ms):
        if landmarks is None:
            return
        # ランドマーク取得
        lw = self._get_lm(landmarks, mp.solutions.pose.PoseLandmark.LEFT_WRIST)
        le = self._get_lm(landmarks, mp.solutions.pose.PoseLandmark.LEFT_ELBOW)
        rw = self._get_lm(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_WRIST)
        re = self._get_lm(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW)

        left_cond = self._is_vertical_up(lw, le)
        right_cond = self._is_vertical_up(rw, re)

        # 左
        if left_cond:
            self.left_in_ms += dt_ms
            self.left_out_ms = 0
            if not self.left_active and self.left_in_ms >= self.hold_ms:
                # 左手は左手スタミナ＆ロックアウトを参照
                if (not self.left_lockout) and (self.left_stamina_ms > 0):
                    self.left_active = True
                # ガード ON（コンソール出力は抑制）
        else:
            self.left_out_ms += dt_ms
            self.left_in_ms = 0
            if self.left_active and self.left_out_ms >= self.hold_ms:
                self.left_active = False
                # ガード OFF（コンソール出力は抑制）

        # 右
        if right_cond:
            self.right_in_ms += dt_ms
            self.right_out_ms = 0
            if not self.right_active and self.right_in_ms >= self.hold_ms:
                if (not self.right_lockout) and (self.right_stamina_ms > 0):
                    self.right_active = True
                # ガード ON（コンソール出力は抑制）
        else:
            self.right_out_ms += dt_ms
            self.right_in_ms = 0
            if self.right_active and self.right_out_ms >= self.hold_ms:
                self.right_active = False
                # ガード OFF（コンソール出力は抑制）

        # --- Guard Stamina 更新（左右独立） ---
        # 左手
        if self.left_active:
            self.left_stamina_ms -= dt_ms
            if self.left_stamina_ms <= 0:
                self.left_stamina_ms = 0
                self.left_active = False
                self.left_lockout = True
        else:
            if self.left_stamina_ms < self.left_stamina_max_ms:
                self.left_stamina_ms += dt_ms
                if self.left_stamina_ms >= self.left_stamina_max_ms:
                    self.left_stamina_ms = self.left_stamina_max_ms
                    if self.left_lockout:
                        self.left_lockout = False
        # 右手
        if self.right_active:
            self.right_stamina_ms -= dt_ms
            if self.right_stamina_ms <= 0:
                self.right_stamina_ms = 0
                self.right_active = False
                self.right_lockout = True
        else:
            if self.right_stamina_ms < self.right_stamina_max_ms:
                self.right_stamina_ms += dt_ms
                if self.right_stamina_ms >= self.right_stamina_max_ms:
                    self.right_stamina_ms = self.right_stamina_max_ms
                    if self.right_lockout:
                        self.right_lockout = False

# パンチ検出器（肩に対する手首の相対Zの急変でパンチ発火）
class PunchDetector:
    """
    肩に対する手首の相対Z（wrist.z - shoulder.z）の **フレーム間変化量** を肩幅とdtで正規化した
    **前方成分の速度 v** に加えて、手首–肩の2D距離 `dist_ws` の **増加量（肩幅で正規化）** を
    併用する AND 条件でパンチを検出する。

    定義：
        v = ( (prev_rel_z - curr_rel_z) / shoulder_width_2D ) / dt_seconds
        dd_norm = (dist_ws_t - dist_ws_{t-1}) / shoulder_width_2D
    条件： v >= vel_thresh かつ dd_norm >= ext_thresh

    ※ MediaPipeのZはカメラに近いほど負になりやすいため、手前へ素早く突き出すと v は正に大きくなる。
    """
    def __init__(self, player_name="P", owner=None, vel_thresh=0.10, ext_thresh=0.20, cooldown_ms=600):
        self.player_name = player_name
        self.vel_thresh = vel_thresh  # 正規化前方速度しきい値
        self.ext_thresh = ext_thresh  # 正規化した腕伸長増分しきい値（肩幅比）
        self.cooldown_ms = cooldown_ms    # 連続発火を抑えるクールダウン
        self.owner = owner

        self.prev_left_rel_z = None
        self.prev_right_rel_z = None
        self.prev_left_dist_ws = None
        self.prev_right_dist_ws = None
        self.left_cd = 0
        self.right_cd = 0

    def _get(self, landmarks, idx):
        return landmarks.landmark[idx]

    def update(self, landmarks, dt_ms):
        if landmarks is None:
            return
        # クールダウン更新
        if self.left_cd > 0:
            self.left_cd -= dt_ms
        if self.right_cd > 0:
            self.right_cd -= dt_ms

        # dt を秒に（0割防止）
        dt_s = max(dt_ms, 1) / 1000.0

        # 両肩ランドマーク（肩幅でスケール正規化）
        ls = self._get(landmarks, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
        rs = self._get(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
        shoulder_w = math.hypot(ls.x - rs.x, ls.y - rs.y) + 1e-6  # 2D肩幅（正規化座標）

        # 左: 手首と肩
        lw = self._get(landmarks, mp.solutions.pose.PoseLandmark.LEFT_WRIST)
        left_rel_z = lw.z - ls.z
        # 腕伸長（2D距離）
        left_dist_ws = math.hypot(lw.x - ls.x, lw.y - ls.y)
        if self.prev_left_rel_z is not None:
            # 手前（負方向）への移動量を正にする差分
            dz = self.prev_left_rel_z - left_rel_z
            # 肩幅 & dt で正規化した速度
            v = (dz / shoulder_w) / dt_s
            # 伸長の正規化増分（肩幅で割る）
            if self.prev_left_dist_ws is not None:
                dd_norm = (left_dist_ws - self.prev_left_dist_ws) / shoulder_w
            else:
                dd_norm = 0.0
            if v >= self.vel_thresh and dd_norm >= self.ext_thresh and self.left_cd <= 0:
                # パンチ中は同じ手のガードを強制解除
                if self.owner is not None and getattr(self.owner, "guard_detector", None) is not None:
                    gd = self.owner.guard_detector
                    gd.left_active = False
                    gd.left_in_ms = 0
                    gd.left_out_ms = gd.hold_ms  # すぐ再ONにならないようにホールド分加算
                # 左パンチ: 相手の左ガードで防がれる
                opp = self.owner.opponent if (self.owner is not None and getattr(self.owner, "opponent", None) is not None) else None
                blocked = False
                if opp is not None and getattr(opp, "guard_detector", None) is not None:
                    blocked = bool(opp.guard_detector.left_active)
                if blocked:
                    print(f"{self.player_name}: 左パンチはガードに防がれた v={v:.2f} dd={dd_norm:.2f}")
                    SoundManager.play("guard")
                    try:
                        if opp is not None and hasattr(opp, 'guard_block_event'):
                            opp.guard_block_event['left'] = pygame.time.get_ticks()
                    except Exception:
                        pass
                    # ガード成功時は攻撃側が反動ダメージ（パンチの半分）
                    try:
                        if self.owner is not None:
                            half_dmg = max(1, int(self.owner.attack_power * 0.5))
                            self.owner.take_reflect_damage(amount=half_dmg)
                    except Exception:
                        pass
                else:
                    print(f"{self.player_name}: 左パンチ！ v={v:.2f} dd={dd_norm:.2f}")
                    opp.take_damage(hand='left') if opp is not None else None
                self.left_cd = self.cooldown_ms
        self.prev_left_rel_z = left_rel_z
        self.prev_left_dist_ws = left_dist_ws

        # 右: 手首と肩
        rw = self._get(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_WRIST)
        right_rel_z = rw.z - rs.z
        right_dist_ws = math.hypot(rw.x - rs.x, rw.y - rs.y)
        if self.prev_right_rel_z is not None:
            dz = self.prev_right_rel_z - right_rel_z
            v = (dz / shoulder_w) / dt_s
            if self.prev_right_dist_ws is not None:
                dd_norm = (right_dist_ws - self.prev_right_dist_ws) / shoulder_w
            else:
                dd_norm = 0.0
            if v >= self.vel_thresh and dd_norm >= self.ext_thresh and self.right_cd <= 0:
                # パンチ中は同じ手のガードを強制解除
                if self.owner is not None and getattr(self.owner, "guard_detector", None) is not None:
                    gd = self.owner.guard_detector
                    gd.right_active = False
                    gd.right_in_ms = 0
                    gd.right_out_ms = gd.hold_ms  # すぐ再ONにならないようにホールド分加算
                # 右パンチ: 相手の右ガードで防がれる
                opp = self.owner.opponent if (self.owner is not None and getattr(self.owner, "opponent", None) is not None) else None
                blocked = False
                if opp is not None and getattr(opp, "guard_detector", None) is not None:
                    blocked = bool(opp.guard_detector.right_active)
                if blocked:
                    print(f"{self.player_name}: 右パンチはガードに防がれた v={v:.2f} dd={dd_norm:.2f}")
                    SoundManager.play("guard")
                    try:
                        if opp is not None and hasattr(opp, 'guard_block_event'):
                            opp.guard_block_event['right'] = pygame.time.get_ticks()
                    except Exception:
                        pass
                    # ガード成功時は攻撃側が反動ダメージ（パンチの半分）
                    try:
                        if self.owner is not None:
                            half_dmg = max(1, int(self.owner.attack_power * 0.5))
                            self.owner.take_reflect_damage(amount=half_dmg)
                    except Exception:
                        pass
                else:
                    print(f"{self.player_name}: 右パンチ！ v={v:.2f} dd={dd_norm:.2f}")
                    opp.take_damage(hand='right') if opp is not None else None
                self.right_cd = self.cooldown_ms
        self.prev_right_rel_z = right_rel_z
        self.prev_right_dist_ws = right_dist_ws

# 将来的にポーズ→アクションに変換するクラス（今はダミーの判定を返すだけ）
class ActionRecognizer:
    def recognize(self, landmarks):
        # TODO: 肩・肘・手首の角度などから punch/kick/jump などを判定
        # 初期段階ではまだ何も返さない
        return None

class Game:
    def __init__(self):
        pygame.init()
        SoundManager.init(base_dir="./sounds")
        SoundManager.load("beam", "beam.mp3", volume=1.0)
        SoundManager.load("guard", "Guard.mp3", volume=10.0)
        # 2台のカメラを設定
        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(1)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("カメラが開けませんでした。")
            exit()
        
        # 画面サイズをHD解像度 (1280x720) に設定
        self.frame_width = 1280
        self.frame_height = 720
        # 各プレイヤーの表示エリア幅（画面を左右に二分）
        self.area_width = self.frame_width // 2
        # 表示用のスケール/クロップ（毎フレーム更新）
        self.w_scaled1 = self.w_scaled2 = 0
        self.h_scaled1 = self.h_scaled2 = 0
        self.crop_x1 = self.crop_x2 = 0
        
        # プレイヤー1、2の初期設定
        self.player1 = Player(health=300, attack_power=10, name="P1")
        self.player2 = Player(health=300, attack_power=10, name="P2")
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1

        # 骨格表示のON/OFFフラグ
        self.show_pose = True  # Mediapipe骨格表示のON/OFF

        # Pose 検出器（カメラごとに1つ）とアクション認識器
        self.pose1 = PoseDetector(enable_segmentation=True, draw_landmarks=self.show_pose)
        self.pose2 = PoseDetector(enable_segmentation=True, draw_landmarks=self.show_pose)
        self.action_recognizer = ActionRecognizer()

        # 1つのウィンドウに2つのカメラ映像を表示するための画面設定
        self.screen = pygame.display.set_mode((self.frame_width, self.frame_height))
        pygame.display.set_caption("2人プレイヤー - 2台のカメラ映像")
        # 背景画像の読み込み（全体に1枚を敷く）
        self.bg_surf = None
        try:
            bg = pygame.image.load("./images/background.PNG")
            # 透過付きPNGなら convert_alpha、そうでなければ convert
            bg = bg.convert_alpha() if bg.get_alpha() else bg.convert()
            self.bg_surf = pygame.transform.smoothscale(bg, (self.frame_width, self.frame_height))
        except Exception as e:
            print(f"背景画像の読み込みに失敗しました: {e}")
            self.bg_surf = None

        # A.T.フィールド画像とスケール係数（肩幅基準）
        self.atfield_img = None
        self.atfield_scale = 2.0  # 肩幅[px] × 係数 が画像の一辺になる（2倍）
        try:
            img = pygame.image.load("./images/atfield.PNG")
            self.atfield_img = img.convert_alpha() if img.get_alpha() else img.convert()
        except Exception as e:
            print(f"A.T.フィールド画像の読み込みに失敗しました: {e}")
            self.atfield_img = None

        # ガード成功時のATフィールド色相反転（180°）の表示時間(ms)
        self.atfield_block_hue_ms = 500
        # 180度色相変更版のATフィールド画像を事前生成
        self.atfield_img_hue180 = None
        try:
            if self.atfield_img is not None:
                self.atfield_img_hue180 = self._make_hue_shifted(self.atfield_img, 180)
        except Exception as e:
            print(f"ATフィールドの色相変換に失敗しました: {e}")

        # 攻撃ヒット時の肩エフェクト（flame）設定
        self.hit_img = None
        self.hit_scale = 2.0   # 肩幅[px] × 係数 が画像一辺（正方形）
        self.hit_ms = 1000     # 表示時間(ms)
        try:
            flame = pygame.image.load("./images/flame.png")
            self.hit_img = flame.convert_alpha() if flame.get_alpha() else flame.convert()
        except Exception as e:
            print(f"ヒット画像の読み込みに失敗しました: {e}")
            self.hit_img = None

        # 経過時間管理（ミリ秒）
        self.prev_ticks = pygame.time.get_ticks()
        # フォント
        pygame.font.init()
        self.ui_font = pygame.font.SysFont(None, 22)
        # 反射ダメージ時の赤枠表示設定
        self.reflect_border_ms = 500
        # おおよそ1cm程度の幅（解像度720p基準で約28px）。必要なら調整可。
        self.reflect_border_px = max(8, int(self.frame_height * 0.04))
        self.reflect_border_color = (255, 0, 0)
    def _draw_reflect_border_for_player(self, screen, player, player_idx):
        ev = getattr(player, 'reflect_event', None)
        if not ev:
            return
        try:
            now = pygame.time.get_ticks()
            if now - ev.get('t0', 0) > ev.get('dur', self.reflect_border_ms):
                player.reflect_event = None
                return
        except Exception:
            player.reflect_event = None
            return
        # プレイヤーごとの描画領域
        if player_idx == 1:
            area_rect = pygame.Rect(0, 0, self.area_width, self.frame_height)
        else:
            area_rect = pygame.Rect(self.area_width, 0, self.area_width, self.frame_height)
        bw = int(getattr(self, 'reflect_border_px', 28))
        col = getattr(self, 'reflect_border_color', (255, 0, 0))
        # 上下左右の枠を描く
        try:
            # 上
            pygame.draw.rect(screen, col, pygame.Rect(area_rect.left, area_rect.top, area_rect.width, bw))
            # 下
            pygame.draw.rect(screen, col, pygame.Rect(area_rect.left, area_rect.bottom - bw, area_rect.width, bw))
            # 左
            pygame.draw.rect(screen, col, pygame.Rect(area_rect.left, area_rect.top, bw, area_rect.height))
            # 右
            pygame.draw.rect(screen, col, pygame.Rect(area_rect.right - bw, area_rect.top, bw, area_rect.height))
        except Exception:
            pass
    def _draw_hit_effect_for_player(self, screen, player, player_idx):
        """攻撃が成功したとき、被弾側(player)の肩に flame 画像を一定時間貼る。
        画像サイズは肩幅(px)×self.hit_scale。self.hit_ms 経過で消える。
        """
        if self.hit_img is None:
            return
        ev = getattr(player, 'hit_event', None)
        if not ev:
            return
        # 時間経過チェック
        now = pygame.time.get_ticks()
        if now - ev.get('t0', 0) >= getattr(self, 'hit_ms', 1000):
            player.hit_event = None
            return
        # どちらの肩か
        shoulder_side = ev.get('shoulder')
        if shoulder_side not in ('left', 'right'):
            return
        # ランドマークが必要
        if player.pose_landmarks is None:
            return
        try:
            # 両肩ランドマーク
            ls = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
            rs = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            ls_xy = self._landmark_to_display_xy(ls, player_idx)
            rs_xy = self._landmark_to_display_xy(rs, player_idx)

            # 表示位置: 2つの肩の間を4等分し、端の次の位置に少しずらす
            # 左肩なら 1/4（左端の次）、右肩なら 3/4（右端の次）
            t = 0.75 if shoulder_side == 'left' else 0.25  # 鏡映のため左右反転
            tx = int(ls_xy[0] + (rs_xy[0] - ls_xy[0]) * t)
            ty = int(ls_xy[1] + (rs_xy[1] - ls_xy[1]) * t)

            # 肩幅(px)を算出してサイズ決定（self.hit_scale は肩幅比）
            shoulder_w_px = max(1, int(math.hypot(ls_xy[0] - rs_xy[0], ls_xy[1] - rs_xy[1])))
            size = max(8, int(self.hit_scale * shoulder_w_px))
            img_scaled = pygame.transform.smoothscale(self.hit_img, (size, size))
            rect = img_scaled.get_rect(center=(tx, ty))
            # プレイヤーの描画領域にクリップ
            if player_idx == 1:
                area_rect = pygame.Rect(0, 0, self.area_width, self.frame_height)
            else:
                area_rect = pygame.Rect(self.area_width, 0, self.area_width, self.frame_height)
            prev_clip = screen.get_clip()
            screen.set_clip(area_rect)
            try:
                screen.blit(img_scaled, rect)
            finally:
                screen.set_clip(prev_clip)
        except Exception:
            return

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # 'P'キーで骨格表示をトグル
                    self.show_pose = not self.show_pose
                    # 2つのPoseDetectorに反映
                    if hasattr(self, 'pose1') and self.pose1 is not None:
                        self.pose1.draw_landmarks = self.show_pose
                    if hasattr(self, 'pose2') and self.pose2 is not None:
                        self.pose2.draw_landmarks = self.show_pose
        return True
    
    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:  # プレイヤー1 - aキー
            self.player1.take_damage()
        if keys[pygame.K_b]:  # プレイヤー2 - bキー
            self.player2.take_damage()

    def update(self):
        # 経過時間（ms）を算出
        current_ticks = pygame.time.get_ticks()
        dt_ms = current_ticks - self.prev_ticks
        self.prev_ticks = current_ticks

        # カメラ1からフレームを取得
        ret1, frame1 = self.cap1.read()
        if not ret1:
            print("カメラ1からフレームを取得できませんでした。")
            return True
        # 入力直後に左右反転（ミラー表示）
        frame1 = cv2.flip(frame1, 1)

        # カメラ2からフレームを取得
        ret2, frame2 = self.cap2.read()
        if not ret2:
            print("カメラ2からフレームを取得できませんでした。")
            return True
        # 入力直後に左右反転（ミラー表示）
        frame2 = cv2.flip(frame2, 1)
        
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1_person_rgba, landmarks1 = self.pose1.process(frame1_rgb)

        # 表示用：高さをエリア高に一致させる（左右ははみ出し可。はみ出しはクロップ）
        h1, w1 = frame1_person_rgba.shape[:2]
        scale1 = self.frame_height / h1
        new_width1 = int(w1 * scale1)
        new_height1 = self.frame_height
        self.w_scaled1, self.h_scaled1 = new_width1, new_height1
        # はみ出す分は中央からクロップ
        self.crop_x1 = max((new_width1 - self.area_width) // 2, 0)
        frame1_disp = cv2.resize(frame1_person_rgba, (new_width1, new_height1), interpolation=cv2.INTER_LINEAR)
        frame1_disp = np.ascontiguousarray(frame1_disp)
        frame1_surface = pygame.image.frombuffer(frame1_disp.tobytes(), (new_width1, new_height1), 'RGBA').convert_alpha()

        # プレイヤーへランドマークを渡して将来のアクション判定に利用
        self.player1.set_pose_landmarks(landmarks1)
        self.player1.update_action(self.action_recognizer)
        self.player1.update_guard(dt_ms)
        self.player1.update_punch(dt_ms)
        
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame2_person_rgba, landmarks2 = self.pose2.process(frame2_rgb)

        # 表示用：高さをエリア高に一致させる（左右ははみ出し可。はみ出しはクロップ）
        h2, w2 = frame2_person_rgba.shape[:2]
        scale2 = self.frame_height / h2
        new_width2 = int(w2 * scale2)
        new_height2 = self.frame_height
        self.w_scaled2, self.h_scaled2 = new_width2, new_height2
        self.crop_x2 = max((new_width2 - self.area_width) // 2, 0)
        frame2_disp = cv2.resize(frame2_person_rgba, (new_width2, new_height2), interpolation=cv2.INTER_LINEAR)
        frame2_disp = np.ascontiguousarray(frame2_disp)
        frame2_surface = pygame.image.frombuffer(frame2_disp.tobytes(), (new_width2, new_height2), 'RGBA').convert_alpha()

        self.player2.set_pose_landmarks(landmarks2)
        self.player2.update_action(self.action_recognizer)
        self.player2.update_guard(dt_ms)
        self.player2.update_punch(dt_ms)
        
        # 背景を描画（全体に1枚）
        if getattr(self, "bg_surf", None) is not None:
            self.screen.blit(self.bg_surf, (0, 0))
        else:
            self.screen.fill((0, 0, 0))
        
        # 左側にカメラ1の映像を表示（クロップ対応）
        if self.w_scaled1 > self.area_width:
            src_rect1 = pygame.Rect(self.crop_x1, 0, self.area_width, self.frame_height)
            self.screen.blit(frame1_surface, (0, 0), src_rect1)
        else:
            # 幅が足りない場合は左寄せ（中央寄せにしたい場合はオフセット追加）
            self.screen.blit(frame1_surface, (0, 0))
        self.player1.update_flash()
        self._draw_reflect_border_for_player(self.screen, self.player1, player_idx=1)
        # ガード中の手の中点にATフィールドを描画
        self._draw_atfield_for_player(self.screen, self.player1, player_idx=1)
        # 画面全体が赤色になる機能は一時停止（コメントアウト）
        # if self.player1.damage_flash:
        #     flash_surface = pygame.Surface((new_width1, self.frame_height), pygame.SRCALPHA)
        #     flash_surface.fill((255, 0, 0, 128))  # 128 = 50%透明度
        #     self.screen.blit(flash_surface, (0, 0))
        self._draw_hit_effect_for_player(self.screen, self.player1, player_idx=1)
        # プレイヤー1の体力ゲージは左側の中央に表示
        self.player1.draw_health_bar(self.screen, self.area_width, self.frame_height)
        self.draw_guard_box_with_offset(self.screen, self.player1, 0, self.area_width, self.frame_height)

        # 右側にカメラ2の映像を表示（クロップ対応）
        if self.w_scaled2 > self.area_width:
            src_rect2 = pygame.Rect(self.crop_x2, 0, self.area_width, self.frame_height)
            self.screen.blit(frame2_surface, (self.area_width, 0), src_rect2)
        else:
            self.screen.blit(frame2_surface, (self.area_width, 0))
        self.player2.update_flash()
        self._draw_reflect_border_for_player(self.screen, self.player2, player_idx=2)
        # ガード中の手の中点にATフィールドを描画
        self._draw_atfield_for_player(self.screen, self.player2, player_idx=2)
        # 画面全体が赤色になる機能は一時停止（コメントアウト）
        # if self.player2.damage_flash:
        #     flash_surface = pygame.Surface((new_width2, self.frame_height), pygame.SRCALPHA)
        #     flash_surface.fill((255, 0, 0, 128))
        #     self.screen.blit(flash_surface, (new_width1, 0))
        self._draw_hit_effect_for_player(self.screen, self.player2, player_idx=2)
        # プレイヤー2の体力ゲージは右側の中央に表示
        self.draw_health_bar_with_offset(self.screen, self.player2, self.area_width, self.area_width, self.frame_height)
        self.draw_guard_box_with_offset(self.screen, self.player2, self.area_width, self.area_width, self.frame_height)

        # 画面を更新

        pygame.display.update()
        
        if self.player1.health <= 0:
            return 1
        elif self.player2.health <= 0:
            return 2

        return False

    def draw_health_bar_with_offset(self, screen, player, offset_x, frame_width, frame_height):
        health_bar_width = frame_width * 0.4
        health_bar_height = 20
        health_bar_x = offset_x + (frame_width - health_bar_width) // 2  # 右側の画面に合わせて中央に配置
        health_bar = pygame.Rect(health_bar_x, 20, health_bar_width, health_bar_height)
        pygame.draw.rect(screen, (255, 0, 0), health_bar)  # 赤い枠
        ratio = 0 if player.max_health <= 0 else max(0.0, min(1.0, player.health / player.max_health))
        pygame.draw.rect(screen, (0, 255, 0), (health_bar_x, 20, ratio * health_bar_width, health_bar_height))  # 緑の体力ゲージ
        # --- HP数値（バー直下中央、フォントはバー高さの約2倍） ---
        try:
            hp_font_size = max(12, int(health_bar_height * 2))
            hp_font = pygame.font.SysFont(None, hp_font_size)
            hp_text = str(int(max(0, player.health)))
            hp_surf = hp_font.render(hp_text, True, (255, 255, 255))
            hp_rect = hp_surf.get_rect()
            hp_rect.midtop = (int(health_bar_x + health_bar_width / 2), int(20 + health_bar_height + 2))
            screen.blit(hp_surf, hp_rect)
        except Exception:
            pass
        # --- Guard Stamina Circles (HPゲージ左右) ---
        def _draw_pie(surface, center, radius, fraction, color):
            if fraction <= 0:
                return
            cx, cy = center
            steps = max(2, int(60 * fraction))
            theta_max = 2 * math.pi * fraction
            pts = [(cx, cy)]
            for i in range(steps + 1):
                t = (i / steps) * theta_max - math.pi / 2
                x = int(cx + radius * math.cos(t))
                y = int(cy + radius * math.sin(t))
                pts.append((x, y))
            try:
                pygame.draw.polygon(surface, color, pts)
            except Exception:
                pass

        gd = getattr(player, 'guard_detector', None)
        if gd is not None:
            radius = int(health_bar_height * 3)  # 1.5倍（従来: 2*h → 3*h）
            cy = 20 + health_bar_height // 2
            pad = 12
            # 左右を逆に配置し、下方向へ半径分オフセット
            left_center = (int(health_bar_x + health_bar_width + pad + radius), int(cy + radius))
            right_center = (int(health_bar_x - pad - radius), int(cy + radius))
            pygame.draw.circle(screen, (80, 80, 80), left_center, radius)
            pygame.draw.circle(screen, (80, 80, 80), right_center, radius)
            lsmax = max(1, int(getattr(gd, 'left_stamina_max_ms', 3000)))
            lsval = max(0, int(getattr(gd, 'left_stamina_ms', 0)))
            lsr = max(0.0, min(1.0, lsval / lsmax))
            l_charging = getattr(gd, 'left_lockout', False) and (lsval < lsmax)
            l_color = (255, 255, 0) if l_charging else (0, 0, 255)
            _draw_pie(screen, left_center, radius, lsr, l_color)
            rsmax = max(1, int(getattr(gd, 'right_stamina_max_ms', 3000)))
            rsval = max(0, int(getattr(gd, 'right_stamina_ms', 0)))
            rsr = max(0.0, min(1.0, rsval / rsmax))
            r_charging = getattr(gd, 'right_lockout', False) and (rsval < rsmax)
            r_color = (255, 255, 0) if r_charging else (0, 0, 255)
            _draw_pie(screen, right_center, radius, rsr, r_color)
            pygame.draw.circle(screen, (200, 200, 200), left_center, radius, 1)
            pygame.draw.circle(screen, (200, 200, 200), right_center, radius, 1)

    def draw_guard_box_with_offset(self, screen, player, offset_x, frame_width, frame_height):
        # 体力ゲージと同程度のサイズ感
        box_width = int(frame_width * 0.4)
        box_height = 20
        margin = 20
        x = offset_x + margin  # 各プレイヤー画面の左寄せ
        y = frame_height - box_height - margin  # 下寄せ

        # 背景（半透明）
        box_surf = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        box_surf.fill((0, 0, 0, 150))
        # 枠線
        pygame.draw.rect(box_surf, (255, 255, 255), pygame.Rect(0, 0, box_width, box_height), 1)

        # テキスト: 左/右ガードの状態
        left_txt = "ON" if player.guard_detector.left_active else "OFF"
        right_txt = "ON" if player.guard_detector.right_active else "OFF"
        text = f"L:{left_txt}  R:{right_txt}"
        text_surf = self.ui_font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect()
        text_rect.centery = box_height // 2
        text_rect.x = 8
        box_surf.blit(text_surf, text_rect)

        # 座標表示（左手首・肘、右手首・肘）
        try:
            lw = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_WRIST)
            le = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_ELBOW)
            rw = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.RIGHT_WRIST)
            re = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW)
            coord_text1 = f"Lw({lw.x:.2f},{lw.y:.2f}) Le({le.x:.2f},{le.y:.2f})"
            coord_text2 = f"Rw({rw.x:.2f},{rw.y:.2f}) Re({re.x:.2f},{re.y:.2f})"
        except Exception:
            coord_text1 = coord_text2 = "No landmarks"

        coord_surf1 = self.ui_font.render(coord_text1, True, (200, 200, 200))
        coord_surf2 = self.ui_font.render(coord_text2, True, (200, 200, 200))
        coord_rect1 = coord_surf1.get_rect()
        coord_rect2 = coord_surf2.get_rect()
        coord_rect1.x = 8
        coord_rect1.y = box_height + 2
        coord_rect2.x = 8
        coord_rect2.y = box_height + 20

        # 画面に配置
        screen.blit(box_surf, (x, y))
        # 座標テキストを重ねる
        screen.blit(coord_surf1, (x + coord_rect1.x, y + coord_rect1.y))
        screen.blit(coord_surf2, (x + coord_rect2.x, y + coord_rect2.y))

    def _landmark_to_display_xy(self, lm, player_idx):
        """landmark正規化座標(lm.x,lm.y)を、実表示の画面座標へ変換。
        高さフィット（frame_height）＆左右クロップ（中央）前提。
        """
        if player_idx == 1:
            offset_x = 0
            width_scaled = self.w_scaled1
            crop_x = self.crop_x1
        else:
            offset_x = self.area_width
            width_scaled = self.w_scaled2
            crop_x = self.crop_x2
        # 高さは常に frame_height に一致
        x_scaled = lm.x * width_scaled
        y_scaled = lm.y * self.frame_height
        # クロップ分を引き、描画エリアのオフセットを足す
        x_pix = int(offset_x + x_scaled - crop_x)
        y_pix = int(y_scaled)
        return x_pix, y_pix

    def _make_hue_shifted(self, surf, degrees):
        """pygame.Surface(surf) の色相を degrees 度シフトした Surface を返す。
        アルファは保持。OpenCVのHSV(H:0-179)を用いるため、180度は+90に相当。
        """
        try:
            if surf is None:
                return None
            # 画像をRGB配列に（形状は (w,h,3) ）
            arr_rgb = pygame.surfarray.array3d(surf)
            # OpenCV は (h,w,3) を期待するので転置
            arr_rgb = np.transpose(arr_rgb, (1, 0, 2)).copy()
            # RGBA のアルファは別取得
            alpha = None
            if surf.get_masks()[3] != 0 or surf.get_bitsize() in (32,):
                try:
                    alpha = pygame.surfarray.array_alpha(surf)
                    alpha = np.transpose(alpha, (1, 0)).copy()
                except Exception:
                    alpha = None
            # RGB->HSV (OpenCVはH:0-179)
            hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
            shift = int((degrees / 360.0) * 180) % 180
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180
            rgb_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # pygame.Surface へ戻す（再び (w,h,3) に転置）
            rgb_shifted = np.transpose(rgb_shifted, (1, 0, 2))
            out = pygame.Surface(surf.get_size(), pygame.SRCALPHA, 32)
            pygame.surfarray.blit_array(out, rgb_shifted)
            if alpha is not None:
                alpha = np.transpose(alpha, (1, 0))  # (w,h) に戻す
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)
                out.lock()
                try:
                    # アルファを書き戻す
                    pygame.surfarray.pixels_alpha(out)[:, :] = alpha
                finally:
                    out.unlock()
            return out.convert_alpha() if out.get_alpha() else out.convert()
        except Exception as e:
            print(f"[make_hue_shifted] 失敗: {e}")
            return None

    def _draw_atfield_for_player(self, screen, player, player_idx):
        """バリア(ガード)中の手ごとに、手首と肘の中点を中心にATフィールド画像を表示。
        画像サイズは肩幅(px)×self.atfield_scaleで毎フレーム追尾する。
        プレイヤーごとの表示領域にクリッピングして、ATフィールドが他カメラ領域にはみ出さないようにする。
        """
        if self.atfield_img is None:
            return
        lms = getattr(player, 'pose_landmarks', None)
        gd = getattr(player, 'guard_detector', None)
        if lms is None or gd is None:
            return
        try:
            # 肩幅(px)算出
            ls = player.guard_detector._get_lm(lms, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
            rs = player.guard_detector._get_lm(lms, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            ls_xy = self._landmark_to_display_xy(ls, player_idx)
            rs_xy = self._landmark_to_display_xy(rs, player_idx)
            shoulder_w_px = max(1, int(math.hypot(ls_xy[0] - rs_xy[0], ls_xy[1] - rs_xy[1])))
        except Exception:
            return

        # --- プレイヤーごとの表示領域にクリッピング ---
        if player_idx == 1:
            area_rect = pygame.Rect(0, 0, self.area_width, self.frame_height)
        else:
            area_rect = pygame.Rect(self.area_width, 0, self.area_width, self.frame_height)
        prev_clip = screen.get_clip()
        screen.set_clip(area_rect)
        try:
            def _hand_center_and_draw(wrist_idx, elbow_idx, hand_side):
                try:
                    w = player.guard_detector._get_lm(lms, wrist_idx)
                    e = player.guard_detector._get_lm(lms, elbow_idx)
                    wx, wy = self._landmark_to_display_xy(w, player_idx)
                    ex, ey = self._landmark_to_display_xy(e, player_idx)
                    cx = (wx + ex) // 2
                    cy = (wy + ey) // 2
                    size = max(8, int(self.atfield_scale * shoulder_w_px))

                    # ガード成功直後のハイライト期間中は色相180°版を使用
                    use_img = self.atfield_img
                    try:
                        evmap = getattr(player, 'guard_block_event', None)
                        if evmap is not None:
                            t0 = evmap.get(hand_side)
                            if t0 is not None:
                                if pygame.time.get_ticks() - t0 < getattr(self, 'atfield_block_hue_ms', 500):
                                    if getattr(self, 'atfield_img_hue180', None) is not None:
                                        use_img = self.atfield_img_hue180
                    except Exception:
                        pass

                    img_scaled = pygame.transform.smoothscale(use_img, (size, size))
                    rect = img_scaled.get_rect(center=(cx, cy))
                    screen.blit(img_scaled, rect)
                except Exception:
                    pass

            # 左手ガード中
            if getattr(gd, 'left_active', False):
                _hand_center_and_draw(mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, 'left')
            # 右手ガード中
            if getattr(gd, 'right_active', False):
                _hand_center_and_draw(mp.solutions.pose.PoseLandmark.RIGHT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, 'right')
        finally:
            # クリッピングを元に戻す
            screen.set_clip(prev_clip)

    def _draw_laser_for_player(self, screen, player, player_idx):
        """被弾側(player)の画面にレーザーを描画。色は赤、線分を時間的に伸ばし→消す。
        時間構成: pre(0.1s) で0→フル長、main(0.25s) で手前から消える、post(0.1s) は非表示。
        """
        if getattr(player, 'laser_event', None) is None:
            return
        ev = player.laser_event
        hand = ev.get('hand')
        if hand not in ('left', 'right'):
            return
        # ランドマークが無ければ描けない
        if player.pose_landmarks is None:
            return
        now = pygame.time.get_ticks()
        elapsed = now - ev['t0']
        pre_ms = ev['pre_ms']
        main_ms = ev['main_ms']
        post_ms = ev['post_ms']
        total = pre_ms + main_ms + post_ms
        if elapsed >= total:
            player.laser_event = None
            return

        # 画面下辺の5等分：左中間=0.1W, 右中間=0.9W
        if player_idx == 1:
            offset_x, W, H = 0, self.area_width, self.frame_height
        else:
            offset_x, W, H = self.area_width, self.area_width, self.frame_height
        left_mid = (offset_x + int(0.1 * W), H - 1)
        right_mid = (offset_x + int(0.9 * W), H - 1)

        # 肩ランドマーク
        if hand == 'right':
            lm = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            base_pt = right_mid
        else:  # 'left'
            lm = player.guard_detector._get_lm(player.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
            base_pt = left_mid
        shoulder_pt = self._landmark_to_display_xy(lm, player_idx)

        # パラメトリックに線分 [start, end] を描く
        if elapsed < pre_ms:
            # 0→フル長に伸びる
            r = max(0.0, min(1.0, elapsed / pre_ms))
            start_t, end_t = 0.0, r
        elif elapsed < pre_ms + main_ms:
            # 手前（基点）から消えていく
            r = (elapsed - pre_ms) / main_ms
            r = max(0.0, min(1.0, r))
            start_t, end_t = r, 1.0
        else:
            # 後段は非表示（フェード時間は0.25sに含めない指定のため）
            return

        # t∈[0,1]で補間
        def lerp(p0, p1, t):
            return (int(p0[0] + (p1[0] - p0[0]) * t), int(p0[1] + (p1[1] - p0[1]) * t))

        p_start = lerp(base_pt, shoulder_pt, start_t)
        p_end = lerp(base_pt, shoulder_pt, end_t)
        try:
            pygame.draw.line(screen, (255, 0, 0), p_start, p_end, 4)
        except Exception:
            pass

    def run(self):
        running = True
        face = None
        while running:
            running = self.handle_events()
            if not running:
                break
            self.handle_input()
            # update() は副作用があるため一度だけ呼ぶ
            res = self.update()
            if res == 1:
                face = 0
                break
            elif res == 2:
                face = 1
                break
            

        
        # リソース解放
        self.pose1.close()
        self.pose2.close()
        self.cap1.release()
        self.cap2.release()
        pygame.quit()
        
        return face



    # ゲームを開始
game = Game()
face = game.run()

if face is None:
    print("No winner determined; skipping face detection.")
else:
    # Mediapipe Face Detection モジュールの初期化
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # カメラ入力（または画像ファイルも可）
    cap = cv2.VideoCapture(face)
    if not cap.isOpened():
        print(f"Camera {face} could not be opened.")
    else:
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 画像をRGBに変換（MediapipeはRGB入力を想定）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                # 可視化: 検出結果があれば顔を切り抜いてLOSERウィンドウに表示
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        # Mediapipeは相対座標なのでピクセル単位に変換
                        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                        # 顔部分を切り取り（範囲外エラー防止のためクリップ）
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(w, x + w_box), min(h, y + h_box)
                        face_crop = frame[y1:y2, x1:x2]
                        # 顔部分を別ウィンドウで表示
                        if face_crop.size > 0:
                            # 顔画像を5倍に拡大
                            face_crop = cv2.resize(face_crop, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_LINEAR)
                            cv2.putText(
                                face_crop,
                                "LOSER!!",
                                (100, 100),
                                cv2.FONT_HERSHEY_DUPLEX,
                                3,
                                (0, 0, 255),
                                5,
                                cv2.LINE_AA
                            )
                            cv2.imshow("LOSER", face_crop)

                # GUIイベントを処理するために waitKey を呼ぶ（これがないとウィンドウが表示されない）
                key = cv2.waitKey(1) & 0xFF
                # 'q' キーで終了
                if key == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
