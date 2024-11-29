import cv2


class TemplateMatchingTracker:
    def __init__(self, n=3):
        self.initialized = False
        self.n = n
        self.corners = []  # Class variable to store corners

    def init(self, frame, roi):
        x, y, w, h = map(int, roi)
        self.template = frame[y : y + h, x : x + w].copy()
        self.template_width = w
        self.template_height = h
        self.last_position = (x, y, w, h)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call `init` first.")

        x, y, w, h = self.last_position
        search_width = w * self.n
        search_height = h * self.n

        frame_height, frame_width = frame.shape[:2]
        search_x = max(0, x - (search_width - w) // 2)
        search_y = max(0, y - (search_height - h) // 2)
        search_x_end = min(frame_width, search_x + search_width)
        search_y_end = min(frame_height, search_y + search_height)

        search_area = frame[search_y:search_y_end, search_x:search_x_end]

        result = cv2.matchTemplate(
            search_area,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Adjust the top-left corner of the match to the frame coordinates
        top_left = (search_x + max_loc[0], search_y + max_loc[1])
        x, y = top_left
        w, h = self.template_width, self.template_height

        self.last_position = (x, y, w, h)

        # Define the corners of the bounding box
        self.corners = [
            (x, y),  # top_left
            (x + w, y),  # top_right
            (x, y + h),  # bottom_left
            (x + w, y + h),  # bottom_right
        ]

        return True, (x, y, w, h)


class TemplateMatchingScaled(TemplateMatchingTracker):
    def __init__(self, n=3, scale_factors=None):
        super().__init__(n=n)
        self.scale_factors = (
            scale_factors if scale_factors else [0.8, 0.9, 1.0, 1.1, 1.2]
        )

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call `init` first.")

        x, y, w, h = self.last_position
        search_width = w * self.n
        search_height = h * self.n

        frame_height, frame_width = frame.shape[:2]
        search_x = max(0, x - (search_width - w) // 2)
        search_y = max(0, y - (search_height - h) // 2)
        search_x_end = min(frame_width, search_x + search_width)
        search_y_end = min(frame_height, search_y + search_height)

        search_area = frame[search_y:search_y_end, search_x:search_x_end]

        best_match_val = -1
        best_match_pos = None
        best_scale = 1.0

        for scale in self.scale_factors:
            scaled_template = cv2.resize(
                self.template,
                (
                    int(self.template_width * scale),
                    int(self.template_height * scale),
                ),
            )
            result = cv2.matchTemplate(
                search_area, scaled_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_pos = max_loc
                best_scale = scale

        if best_match_pos is None:
            return False, (x, y, w, h)

        top_left = (search_x + best_match_pos[0], search_y + best_match_pos[1])
        x, y = top_left
        w = int(self.template_width * best_scale)
        h = int(self.template_height * best_scale)

        self.last_position = (x, y, w, h)
        self.template = cv2.resize(
            frame[y : y + h, x : x + w],
            (self.template_width, self.template_height),
        )
        self.corners = [
            (x, y),  # top_left
            (x + w, y),  # top_right
            (x, y + h),  # bottom_left
            (x + w, y + h),  # bottom_right
        ]

        return True, (x, y, w, h)
