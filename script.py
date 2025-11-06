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
        self.laser_event = None  # {'hand': 'left'|'right', 't0': ms, 'pre_ms':100, 'main_ms':250, 'post_ms':100}
    
    def take_damage(self, hand=None):
        SoundManager.play("beam")
        self.health -= self.attack_power
        # フラッシュは一旦コメントアウト（仕様により画面全体の赤点滅は停止）
        # self.damage_flash = True
        # self.flash_timer = self.flash_duration
        # レーザーエフェクトを開始（被弾側に描画）
        try:
            t0 = pygame.time.get_ticks()
        except Exception:
            t0 = 0
        self.laser_event = {
            'hand': hand,          # 'left' or 'right'
            't0': t0,
            'pre_ms': 100,         # 前段 0.1s（0.25sには含めない）
            'main_ms': 250,        # 本描画 0.25s
            'post_ms': 100         # 後段 0.1s（0.25sには含めない）
        }
    
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
        pygame.draw.rect(screen, (0, 255, 0), (health_bar_x, 20, (self.health / 100) * health_bar_width, health_bar_height))  # 緑の体力ゲージ

# Mediapipe Pose をラップする検出クラス（1カメラ=1インスタンス推奨）
class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
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

    def process(self, frame_rgb):
        """
        入力: RGBの画像(numpy array)
        出力: (骨格描画済みRGB画像, landmarks)
        """
        results = self.pose.process(frame_rgb)
        landmarks = results.pose_landmarks

        # デフォルトは元フレーム
        output_rgb = frame_rgb

        # セグメンテーションが有効なら人物以外を緑で塗りつぶす
        seg_mask = getattr(results, "segmentation_mask", None)
        if seg_mask is not None:
            # seg_mask は [H, W] のfloat32 (人物らしさ: 0..1)
            cond = seg_mask > 0.5  # しきい値は必要に応じて調整
            green_bg = np.zeros_like(frame_rgb)
            green_bg[:] = (0, 255, 0)  # RGB の緑
            # cond を各チャンネルへ拡張して合成
            output_rgb = np.where(cond[..., None], frame_rgb, green_bg)

        # ランドマーク描画は合成後のフレームに行う
        if landmarks is not None:
            self.mp_drawing.draw_landmarks(
                output_rgb,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        return output_rgb, landmarks

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
                self.right_active = True
                # ガード ON（コンソール出力は抑制）
        else:
            self.right_out_ms += dt_ms
            self.right_in_ms = 0
            if self.right_active and self.right_out_ms >= self.hold_ms:
                self.right_active = False
                # ガード OFF（コンソール出力は抑制）

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
    def __init__(self, player_name="P", owner=None, vel_thresh=0.30, ext_thresh=0.16, cooldown_ms=1000):
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
                print(f"{self.player_name}: 左パンチ！ v={v:.2f} dd={dd_norm:.2f}")
                if self.owner is not None and getattr(self.owner, "opponent", None) is not None:
                    self.owner.opponent.take_damage(hand='left')
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
                print(f"{self.player_name}: 右パンチ！ v={v:.2f} dd={dd_norm:.2f}")
                if self.owner is not None and getattr(self.owner, "opponent", None) is not None:
                    self.owner.opponent.take_damage(hand='right')
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
        # 2台のカメラを設定
        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(1)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("カメラが開けませんでした。")
            exit()
        
        # 画面サイズをHD解像度 (1280x720) に設定
        self.frame_width = 1280
        self.frame_height = 720
        
        # プレイヤー1、2の初期設定
        self.player1 = Player(health=100, attack_power=10, name="P1")
        self.player2 = Player(health=100, attack_power=10, name="P2")
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1

        # Pose 検出器（カメラごとに1つ）とアクション認識器
        self.pose1 = PoseDetector(enable_segmentation=True)
        self.pose2 = PoseDetector(enable_segmentation=True)
        self.action_recognizer = ActionRecognizer()

        # 1つのウィンドウに2つのカメラ映像を表示するための画面設定
        self.screen = pygame.display.set_mode((self.frame_width, self.frame_height))
        pygame.display.set_caption("2人プレイヤー - 2台のカメラ映像")
        # 経過時間管理（ミリ秒）
        self.prev_ticks = pygame.time.get_ticks()
        # フォント
        pygame.font.init()
        self.ui_font = pygame.font.SysFont(None, 22)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
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
            return False

        # カメラ2からフレームを取得
        ret2, frame2 = self.cap2.read()
        if not ret2:
            print("カメラ2からフレームを取得できませんでした。")
            return False
        
        # MediaPipe処理は回転前（カメラの生フレーム基準）で実施
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1_annotated_rgb, landmarks1 = self.pose1.process(frame1_rgb)

        # 表示用に90度回転（骨格描画済み画像を回転）
        frame1_rotated_annotated = cv2.rotate(frame1_annotated_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 表示用リサイズ（縦横比維持で左半分の幅に）
        h1, w1 = frame1_rotated_annotated.shape[:2]
        aspect_ratio1 = w1 / h1
        new_width1 = self.frame_width // 2
        new_height1 = int(new_width1 / aspect_ratio1)
        self.new_width1, self.new_height1 = new_width1, new_height1
        frame1_disp = cv2.resize(frame1_rotated_annotated, (new_width1, new_height1))

        # プレイヤーへランドマークを渡して将来のアクション判定に利用
        self.player1.set_pose_landmarks(landmarks1)
        self.player1.update_action(self.action_recognizer)
        self.player1.update_guard(dt_ms)
        self.player1.update_punch(dt_ms)

        frame1_surface = pygame.surfarray.make_surface(frame1_disp)
        
        # MediaPipe処理は回転前（カメラの生フレーム基準）で実施
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame2_annotated_rgb, landmarks2 = self.pose2.process(frame2_rgb)

        # 表示用に90度回転（骨格描画済み画像を回転）
        frame2_rotated_annotated = cv2.rotate(frame2_annotated_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 表示用リサイズ（縦横比維持で右半分の幅に）
        h2, w2 = frame2_rotated_annotated.shape[:2]
        aspect_ratio2 = w2 / h2
        new_width2 = self.frame_width // 2
        new_height2 = int(new_width2 / aspect_ratio2)
        self.new_width2, self.new_height2 = new_width2, new_height2
        frame2_disp = cv2.resize(frame2_rotated_annotated, (new_width2, new_height2))

        self.player2.set_pose_landmarks(landmarks2)
        self.player2.update_action(self.action_recognizer)
        self.player2.update_guard(dt_ms)
        self.player2.update_punch(dt_ms)

        frame2_surface = pygame.surfarray.make_surface(frame2_disp)
        
        # 画面を黒でクリア
        self.screen.fill((0,0,0))
        
        # 左側にカメラ1の映像を表示
        self.screen.blit(frame1_surface, (0, 0))
        self.player1.update_flash()
        # 画面全体が赤色になる機能は一時停止（コメントアウト）
        # if self.player1.damage_flash:
        #     flash_surface = pygame.Surface((new_width1, self.frame_height), pygame.SRCALPHA)
        #     flash_surface.fill((255, 0, 0, 128))  # 128 = 50%透明度
        #     self.screen.blit(flash_surface, (0, 0))
        self._draw_laser_for_player(self.screen, self.player1, player_idx=1)
        # プレイヤー1の体力ゲージは左側の中央に表示
        self.player1.draw_health_bar(self.screen, new_width1, self.frame_height)
        self.draw_guard_box_with_offset(self.screen, self.player1, 0, new_width1, self.frame_height)

        # 右側にカメラ2の映像を表示
        self.screen.blit(frame2_surface, (new_width1, 0))
        self.player2.update_flash()
        # 画面全体が赤色になる機能は一時停止（コメントアウト）
        # if self.player2.damage_flash:
        #     flash_surface = pygame.Surface((new_width2, self.frame_height), pygame.SRCALPHA)
        #     flash_surface.fill((255, 0, 0, 128))
        #     self.screen.blit(flash_surface, (new_width1, 0))
        self._draw_laser_for_player(self.screen, self.player2, player_idx=2)
        # プレイヤー2の体力ゲージは右側の中央に表示
        # 右側に表示するために描画位置を調整
        # 一時的に画面を切り取って描画する方法ではなく、draw_health_barの引数を変えて描画位置を調整
        # 体力ゲージのx座標はdraw_health_bar内で中央に配置されるため、画面を右側にずらすためにsurfaceを部分的に描画するか、
        # ここではdraw_health_barの引数をnew_width2にして描画し、surfaceを右側にずらすためにスクリーンの描画位置を調整
        # なので、draw_health_bar内のx座標計算はnew_width2の中央
        # しかしdraw_health_barはscreenに直接描画するため、体力ゲージの位置を右側にずらすためにsurfaceを部分的に描画するのは難しい
        # よって、draw_health_barにscreenを渡すのではなく、右側の部分だけ切り出したsurfaceを作成して描画する方法を取る
        # ここでは簡単に、体力ゲージを右側の画面に合わせて描画するためにオフセットを加えるラッパー関数を作成する
        # もしくはdraw_health_barのx座標計算を変更するために、draw_health_barにoffset_xを追加して対応する
        # ここではoffset_xを追加して対応する
        self.draw_health_bar_with_offset(self.screen, self.player2, new_width1, new_width2, self.frame_height)
        self.draw_guard_box_with_offset(self.screen, self.player2, new_width1, new_width2, self.frame_height)

        # 画面を更新
        pygame.display.update()

        if self.player1.health <= 0 or self.player2.health <= 0:
            print("ゲームオーバー！")
            return False
        return True

    def draw_health_bar_with_offset(self, screen, player, offset_x, frame_width, frame_height):
        health_bar_width = frame_width * 0.4
        health_bar_height = 20
        health_bar_x = offset_x + (frame_width - health_bar_width) // 2  # 右側の画面に合わせて中央に配置
        health_bar = pygame.Rect(health_bar_x, 20, health_bar_width, health_bar_height)
        pygame.draw.rect(screen, (255, 0, 0), health_bar)  # 赤い枠
        pygame.draw.rect(screen, (0, 255, 0), (health_bar_x, 20, (player.health / 100) * health_bar_width, health_bar_height))  # 緑の体力ゲージ

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
        """landmark正規化座標(lm.x,lm.y)を、回転後・リサイズ後・配置後の画面座標へ変換。
        player_idx: 1 (左画面) or 2 (右画面)
        回転は90度CCW、x' = y、y' = 1 - x で近似。
        """
        if player_idx == 1:
            offset_x, width, height = 0, self.new_width1, self.new_height1
        else:
            offset_x, width, height = self.new_width1, self.new_width2, self.new_height2
        x_rot_norm = lm.y
        y_rot_norm = 1.0 - lm.x
        x_pix = offset_x + int(x_rot_norm * width)
        y_pix = int(y_rot_norm * height)
        return x_pix, y_pix

    def _draw_laser_for_player(self, screen, player, player_idx):
        """被弾側(player)の画面にレーザーを描画。色は赤、線分を時間的に伸ばし→消す。
        時間構成: pre(0.1s) で0→フル長、main(0.25s) で手前から消える、post(0.1s) は非表示。
        """
        if player.laser_event is None:
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
            offset_x, W, H = 0, self.new_width1, self.new_height1
        else:
            offset_x, W, H = self.new_width1, self.new_width2, self.new_height2
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
        while running:
            running = self.handle_events()
            if not running:
                break
            self.handle_input()
            if not self.update():
                break
        
        # リソース解放
        self.pose1.close()
        self.pose2.close()
        self.cap1.release()
        self.cap2.release()
        pygame.quit()

# ゲームを開始
game = Game()
game.run()
