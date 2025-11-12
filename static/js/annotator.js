/**
 * LeRobot 标注系统前端逻辑
 * 功能: 视频同步控制、标注状态管理、键盘快捷键、数据持久化
 */

class VideoFrameCalculator {
    constructor(episodeId) {
        this.episodeId = episodeId;
        this.videoFrameInfo = new Map(); // 缓存不同视频的帧信息
        this.apiCallCache = new Map(); // API调用缓存
        this.isLoading = new Set(); // 正在加载的视频键
    }

    /**
     * 获取视频的帧信息（支持缓存）
     */
    async getVideoFrameInfo(videoKey) {
        const cacheKey = `${this.episodeId}_${videoKey}`;

        // 检查缓存
        if (this.videoFrameInfo.has(cacheKey)) {
            return this.videoFrameInfo.get(cacheKey);
        }

        // 避免重复API调用
        if (this.isLoading.has(cacheKey)) {
            // 等待正在进行的请求
            while (this.isLoading.has(cacheKey)) {
                await this.delay(50);
            }
            return this.videoFrameInfo.get(cacheKey);
        }

        this.isLoading.add(cacheKey);

        try {
            const response = await fetch(`/api/episode/${this.episodeId}/video_frame_info/${videoKey}`);
            const data = await response.json();

            if (data.success) {
                // 缓存结果
                this.videoFrameInfo.set(cacheKey, data);
                return data;
            } else {
                console.warn(`获取视频帧信息失败: ${data.error}`);
                return null;
            }
        } catch (error) {
            console.error(`获取视频帧信息失败: ${error.message}`);
            return null;
        } finally {
            this.isLoading.delete(cacheKey);
        }
    }

    /**
     * 基于OpenCV数据和视频当前时间计算精确帧号
     */
    async calculateCurrentFrame(videoElement, videoKey) {
        if (!videoElement || !videoKey) {
            return 0;
        }

        try {
            const frameInfo = await this.getVideoFrameInfo(videoKey);

            if (frameInfo && frameInfo.fps > 0) {
                // 使用OpenCV提供的精确FPS计算
                const currentFrame = Math.floor(videoElement.currentTime * frameInfo.fps);
                return Math.max(0, Math.min(currentFrame, frameInfo.total_frames - 1));
            } else {
                // 降级到原有计算方式
                console.warn(`OpenCV帧信息不可用，降级到基于时间的计算: ${videoKey}`);
                return Math.floor(videoElement.currentTime * 30); // 假设30fps
            }
        } catch (error) {
            console.error(`计算当前帧失败: ${error.message}`);
            // 降级方案
            return Math.floor(videoElement.currentTime * 30);
        }
    }

    /**
     * 获取视频总帧数
     */
    async getTotalFrames(videoKey) {
        try {
            const frameInfo = await this.getVideoFrameInfo(videoKey);
            return frameInfo ? frameInfo.total_frames : 0;
        } catch (error) {
            console.error(`获取总帧数失败: ${error.message}`);
            return 0;
        }
    }

    /**
     * 获取视频真实FPS
     */
    async getVideoFPS(videoKey) {
        try {
            const frameInfo = await this.getVideoFrameInfo(videoKey);
            return frameInfo ? frameInfo.fps : 30.0;
        } catch (error) {
            console.error(`获取视频FPS失败: ${error.message}`);
            return 30.0; // 默认FPS
        }
    }

    /**
     * 帧号转时间（基于OpenCV的精确FPS）
     */
    async frameToTime(frame, videoKey) {
        try {
            const fps = await this.getVideoFPS(videoKey);
            return frame / fps;
        } catch (error) {
            console.error(`帧号转时间失败: ${error.message}`);
            return frame / 30.0; // 降级方案
        }
    }

    /**
     * 时间转帧号（基于OpenCV的精确FPS）
     */
    async timeToFrame(time, videoKey) {
        try {
            const fps = await this.getVideoFPS(videoKey);
            const totalFrames = await this.getTotalFrames(videoKey);
            const frame = Math.floor(time * fps);
            return Math.max(0, Math.min(frame, totalFrames - 1));
        } catch (error) {
            console.error(`时间转帧号失败: ${error.message}`);
            return Math.floor(time * 30); // 降级方案
        }
    }

    /**
     * 批量预加载多个视频的帧信息
     */
    async preloadVideoFrameInfo(videoKeys) {
        const promises = videoKeys.map(videoKey => this.getVideoFrameInfo(videoKey));
        try {
            await Promise.all(promises);
            console.log(`预加载${videoKeys.length}个视频的帧信息完成`);
        } catch (error) {
            console.warn(`预加载视频帧信息失败: ${error.message}`);
        }
    }

    /**
     * 清除指定视频的缓存
     */
    clearCache(videoKey = null) {
        if (videoKey) {
            const cacheKey = `${this.episodeId}_${videoKey}`;
            this.videoFrameInfo.delete(cacheKey);
        } else {
            this.videoFrameInfo.clear();
        }
    }

    /**
     * 延迟工具方法
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * 获取调试信息
     */
    getDebugInfo() {
        return {
            episodeId: this.episodeId,
            cachedVideos: Array.from(this.videoFrameInfo.keys()),
            loadingVideos: Array.from(this.isLoading),
            cacheSize: this.videoFrameInfo.size
        };
    }
}

class VideoSyncController {
    constructor(videos, fps, episodeId = null) {
        this.videos = videos;
        this.fps = fps;
        this.currentTime = 0;
        this.totalTime = 0;
        this.isPlaying = false;

        // 集成VideoFrameCalculator用于精确帧计算
        this.frameCalculator = episodeId ? new VideoFrameCalculator(episodeId) : null;
        this.primaryVideoKey = null; // 主要视频键，用于帧计算

        // 缓存的帧信息
        this._cachedTotalFrames = null;
        this._cachedCurrentFrame = null;
    }

    // 设置主要视频键（用于精确帧计算）
    setPrimaryVideoKey(videoKey) {
        this.primaryVideoKey = videoKey;
        this._cachedTotalFrames = null; // 清除缓存
        this._cachedCurrentFrame = null;
    }

    // 同步播放所有视频
    playAll() {
        this.videos.forEach(video => {
            if (video.paused) {
                video.play().catch(e => console.warn('Video play failed:', e));
            }
        });
        this.isPlaying = true;
    }

    // 同步暂停所有视频
    pauseAll() {
        this.videos.forEach(video => {
            if (!video.paused) {
                video.pause();
            }
        });
        this.isPlaying = false;
    }

    // 同步跳转到指定时间
    seekTo(time) {
        const clampedTime = Math.max(0, Math.min(time, this.totalTime));
        this.videos.forEach(video => {
            if (Math.abs(video.currentTime - clampedTime) > 0.1) {
                video.currentTime = clampedTime;
            }
        });
        this.currentTime = clampedTime;
        this._cachedCurrentFrame = null; // 清除当前帧缓存
    }

    // 相对跳转
    seekRelative(seconds) {
        this.seekTo(this.currentTime + seconds);
    }

    // 单帧步进（改进版，使用精确FPS）
    async stepFrames(frameCount) {
        if (this.frameCalculator && this.primaryVideoKey) {
            try {
                const fps = await this.frameCalculator.getVideoFPS(this.primaryVideoKey);
                const timeStep = frameCount / fps;
                this.seekTo(this.currentTime + timeStep);
                return;
            } catch (error) {
                console.warn('精确帧步进失败，降级到传统方法:', error.message);
            }
        }

        // 降级到原有方法
        const timeStep = frameCount / this.fps;
        this.seekTo(this.currentTime + timeStep);
    }

    // 获取当前帧号（改进版，使用OpenCV精确计算）
    async getCurrentFrame() {
        if (this.frameCalculator && this.primaryVideoKey && this.videos.length > 0) {
            try {
                const primaryVideo = this.videos[0]; // 使用第一个视频作为参考
                const currentFrame = await this.frameCalculator.calculateCurrentFrame(primaryVideo, this.primaryVideoKey);
                this._cachedCurrentFrame = currentFrame;
                return currentFrame;
            } catch (error) {
                console.warn('OpenCV帧计算失败，降级到传统方法:', error.message);
            }
        }

        // 降级到原有方法
        const frame = Math.floor(this.currentTime * this.fps);
        this._cachedCurrentFrame = frame;
        return frame;
    }

    // 获取总帧数（改进版，使用OpenCV精确计算）
    async getTotalFrames() {
        if (this._cachedTotalFrames !== null) {
            return this._cachedTotalFrames;
        }

        if (this.frameCalculator && this.primaryVideoKey) {
            try {
                const totalFrames = await this.frameCalculator.getTotalFrames(this.primaryVideoKey);
                if (totalFrames > 0) {
                    this._cachedTotalFrames = totalFrames;
                    return totalFrames;
                }
            } catch (error) {
                console.warn('OpenCV总帧数获取失败，降级到传统方法:', error.message);
            }
        }

        // 降级到原有方法
        const frames = Math.floor(this.totalTime * this.fps);
        this._cachedTotalFrames = frames;
        return frames;
    }

    // 帧号转时间（改进版，使用精确FPS）
    async frameToTime(frame) {
        if (this.frameCalculator && this.primaryVideoKey) {
            try {
                return await this.frameCalculator.frameToTime(frame, this.primaryVideoKey);
            } catch (error) {
                console.warn('精确帧号转时间失败，降级到传统方法:', error.message);
            }
        }

        // 降级到原有方法
        return frame / this.fps;
    }

    // 时间转帧号（改进版，使用精确FPS）
    async timeToFrame(time) {
        if (this.frameCalculator && this.primaryVideoKey) {
            try {
                return await this.frameCalculator.timeToFrame(time, this.primaryVideoKey);
            } catch (error) {
                console.warn('精确时间转帧号失败，降级到传统方法:', error.message);
            }
        }

        // 降级到原有方法
        return Math.floor(time * this.fps);
    }

    // 更新时间状态
    updateTime(time, duration) {
        this.currentTime = time;
        this.totalTime = duration;
        this._cachedCurrentFrame = null; // 清除当前帧缓存
    }

    // 同步检查 - 确保所有视频时间一致
    syncCheck() {
        if (this.videos.length <= 1) return;

        const targetTime = this.videos[0].currentTime;
        const tolerance = 0.1; // 100ms 容忍度

        this.videos.forEach((video, index) => {
            if (index > 0 && Math.abs(video.currentTime - targetTime) > tolerance) {
                video.currentTime = targetTime;
            }
        });
    }

    // 预加载帧信息（用于性能优化）
    async preloadFrameInfo(videoKeys) {
        if (this.frameCalculator && videoKeys.length > 0) {
            try {
                await this.frameCalculator.preloadVideoFrameInfo(videoKeys);
                console.log('视频帧信息预加载完成');
            } catch (error) {
                console.warn('视频帧信息预加载失败:', error.message);
            }
        }
    }

    // 清除帧计算缓存
    clearFrameCache() {
        this._cachedTotalFrames = null;
        this._cachedCurrentFrame = null;
        if (this.frameCalculator) {
            this.frameCalculator.clearCache();
        }
    }

    // 获取调试信息
    getDebugInfo() {
        const debugInfo = {
            currentTime: this.currentTime,
            totalTime: this.totalTime,
            isPlaying: this.isPlaying,
            primaryVideoKey: this.primaryVideoKey,
            cachedTotalFrames: this._cachedTotalFrames,
            cachedCurrentFrame: this._cachedCurrentFrame,
            hasFrameCalculator: !!this.frameCalculator
        };

        if (this.frameCalculator) {
            debugInfo.frameCalculatorInfo = this.frameCalculator.getDebugInfo();
        }

        return debugInfo;
    }
}

class AnnotationStateMachine {
    constructor() {
        this.state = 'idle'; // idle, selecting_start, selecting_end, editing, saving
        this.callbacks = {};
    }

    // 注册状态变化回调
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }

    // 触发事件
    emit(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => callback(data));
        }
    }

    // 状态转换
    setState(newState, data = null) {
        const oldState = this.state;
        this.state = newState;
        this.emit('stateChange', { from: oldState, to: newState, data });
        this.emit(newState, data);
    }

    // 获取当前状态
    getState() {
        return this.state;
    }

    // 状态检查方法
    isIdle() { return this.state === 'idle'; }
    isSelecting() { return this.state.startsWith('selecting'); }
    isEditing() { return this.state === 'editing'; }
    isSaving() { return this.state === 'saving'; }
}

class SegmentManager {
    constructor() {
        this.segments = [];
        this.currentSegment = this.createEmptySegment();
        this.editingIndex = -1;
    }

    createEmptySegment() {
        return {
            start_frame: null,
            end_frame: null,
            skill: '',
            action_text: ''
        };
    }

    // 添加或更新段落
    upsertSegment(segment) {
        // 数据验证
        const validationError = this.validateSegment(segment);
        if (validationError) {
            throw new Error(validationError);
        }

        // 重叠检查
        const overlapError = this.checkOverlap(segment, this.editingIndex);
        if (overlapError) {
            throw new Error(overlapError);
        }

        const newSegment = {
            start_frame: parseInt(segment.start_frame),
            end_frame: parseInt(segment.end_frame),
            skill: segment.skill,
            action_text: segment.action_text.trim()
        };

        if (this.editingIndex >= 0) {
            // 更新现有段落
            this.segments[this.editingIndex] = newSegment;
            this.editingIndex = -1;
        } else {
            // 添加新段落
            this.segments.push(newSegment);
        }

        this.currentSegment = this.createEmptySegment();
        return newSegment;
    }

    // 删除段落
    deleteSegment(index) {
        if (index >= 0 && index < this.segments.length) {
            return this.segments.splice(index, 1)[0];
        }
        return null;
    }

    // 编辑段落
    editSegment(index) {
        if (index >= 0 && index < this.segments.length) {
            const segment = this.segments[index];
            this.currentSegment = { ...segment };
            this.editingIndex = index;
            return this.currentSegment;
        }
        return null;
    }

    // 清空当前编辑
    clearCurrent() {
        this.currentSegment = this.createEmptySegment();
        this.editingIndex = -1;
    }

    // 数据验证
    validateSegment(segment) {
        if (segment.start_frame === null || segment.start_frame === undefined) {
            return '起始帧不能为空';
        }
        if (segment.end_frame === null || segment.end_frame === undefined) {
            return '结束帧不能为空';
        }
        if (segment.start_frame > segment.end_frame) {
            return '起始帧不能大于结束帧';
        }
        if (segment.start_frame < 0) {
            return '帧号不能为负数';
        }
        if (!segment.skill || !segment.skill.trim()) {
            return '动作类型不能为空';
        }
        if (!segment.action_text || !segment.action_text.trim()) {
            return '动作描述不能为空';
        }
        if (segment.action_text.trim().length < 10) {
            return '动作描述至少需要10个字符';
        }
        if (segment.action_text.trim().length > 500) {
            return '动作描述不能超过500个字符';
        }
        return null;
    }

    // 重叠检查
    checkOverlap(newSegment, editingIndex = -1) {
        const hasOverlap = this.segments.some((segment, index) => {
            if (index === editingIndex) return false;

            // 检查半开区间重叠: [a,b) 和 [c,d) 重叠当且仅当 max(a,c) < min(b,d)
            return Math.max(newSegment.start_frame, segment.start_frame) <
                   Math.min(newSegment.end_frame, segment.end_frame);
        });

        return hasOverlap ? '段落与现有段落重叠，请检查帧范围' : null;
    }

    // 获取所有段落
    getAllSegments() {
        return [...this.segments];
    }

    // 设置段落列表
    setSegments(segments) {
        this.segments = [...(segments || [])];
    }

    // 获取段落统计信息
    getStats() {
        return {
            count: this.segments.length,
            skills: [...new Set(this.segments.map(s => s.skill))],
            totalFrames: this.segments.reduce((sum, s) => sum + (s.end_frame - s.start_frame + 1), 0)
        };
    }
}

class KeyboardHandler {
    constructor(videoController, segmentManager, stateMachine) {
        this.videoController = videoController;
        this.segmentManager = segmentManager;
        this.stateMachine = stateMachine;
        this.callbacks = {};
        this.setupKeyboardEvents();
    }

    // 注册键盘快捷键回调
    on(key, callback) {
        if (!this.callbacks[key]) {
            this.callbacks[key] = [];
        }
        this.callbacks[key].push(callback);
    }

    // 设置键盘事件监听
    setupKeyboardEvents() {
        document.addEventListener('keydown', (e) => {
            // 避免在输入框中触发快捷键
            if (this.isInputElement(e.target)) {
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }

            this.handleKeyDown(e);
        });
    }

    // 检查是否是输入元素
    isInputElement(element) {
        const inputTypes = ['INPUT', 'TEXTAREA', 'SELECT'];
        return inputTypes.includes(element.tagName) || element.contentEditable === 'true';
    }

    // 处理按键事件
    handleKeyDown(e) {
        const key = e.key.toLowerCase();
        const ctrl = e.ctrlKey;
        const shift = e.shiftKey;
        const alt = e.altKey;

        // 基础视频控制
        switch (key) {
            case ' ':
                e.preventDefault();
                this.togglePlayPause();
                break;
            case 'arrowleft':
                e.preventDefault();
                if (shift) {
                    this.videoController.seekRelative(-5); // 快速后退5秒
                } else {
                    this.videoController.stepFrames(-1); // 单帧后退
                }
                break;
            case 'arrowright':
                e.preventDefault();
                if (shift) {
                    this.videoController.seekRelative(5); // 快速前进5秒
                } else {
                    this.videoController.stepFrames(1); // 单帧前进
                }
                break;
            case 'home':
                e.preventDefault();
                this.videoController.seekTo(0);
                break;
            case 'end':
                e.preventDefault();
                this.videoController.seekTo(this.videoController.totalTime);
                break;
        }

        // 标注控制
        switch (key) {
            case 's':
                if (ctrl) {
                    e.preventDefault();
                    this.triggerCallback('save');
                } else {
                    e.preventDefault();
                    this.triggerCallback('setStartFrame');
                }
                break;
            case 'e':
                e.preventDefault();
                this.triggerCallback('setEndFrame');
                break;
            case 'a':
                if (ctrl) {
                    e.preventDefault();
                    // Ctrl+A 全选，不处理
                } else {
                    e.preventDefault();
                    this.triggerCallback('addSegment');
                }
                break;
            case 'delete':
            case 'backspace':
                if (!this.isInputElement(document.activeElement)) {
                    e.preventDefault();
                    this.triggerCallback('deleteSelected');
                }
                break;
            case 'escape':
                e.preventDefault();
                this.triggerCallback('cancel');
                break;
        }

        // Episode导航
        switch (key) {
            case 'arrowup':
                e.preventDefault();
                this.triggerCallback('previousEpisode');
                break;
            case 'arrowdown':
                e.preventDefault();
                this.triggerCallback('nextEpisode');
                break;
        }

        // 视频显示控制
        if (ctrl && /^[1-9]$/.test(key)) {
            e.preventDefault();
            const videoIndex = parseInt(key) - 1;
            this.triggerCallback('toggleVideo', videoIndex);
        }
    }

    // 切换播放/暂停
    togglePlayPause() {
        if (this.videoController.isPlaying) {
            this.videoController.pauseAll();
            this.triggerCallback('pause');
        } else {
            this.videoController.playAll();
            this.triggerCallback('play');
        }
    }

    // 触发回调
    triggerCallback(action, data = null) {
        if (this.callbacks[action]) {
            this.callbacks[action].forEach(callback => callback(data));
        }
    }
}

class APIClient {
    constructor() {
        this.baseUrl = '';
        this.cache = new Map();
        this.retryCount = 3;
        this.retryDelay = 1000;
    }

    // 通用请求方法
    async request(url, options = {}) {
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        let lastError = null;

        for (let i = 0; i < this.retryCount; i++) {
            try {
                const response = await fetch(url, config);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                return data;
            } catch (error) {
                lastError = error;

                if (i < this.retryCount - 1) {
                    await this.delay(this.retryDelay * Math.pow(2, i));
                }
            }
        }

        throw lastError;
    }

    // 延迟工具方法
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // 加载标注数据
    async loadAnnotations(episodeId) {
        const cacheKey = `annotations_${episodeId}`;

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            const data = await this.request(`/api/annotations/${episodeId}`);
            this.cache.set(cacheKey, data);
            return data;
        } catch (error) {
            console.error('Failed to load annotations:', error);
            // 返回默认结构
            return {
                episode_id: episodeId,
                label_info: { action_config: [] }
            };
        }
    }

    // 保存标注数据
    async saveAnnotations(episodeId, data) {
        try {
            const result = await this.request(`/api/annotations/${episodeId}`, {
                method: 'POST',
                body: JSON.stringify(data)
            });

            // 更新缓存
            const cacheKey = `annotations_${episodeId}`;
            this.cache.set(cacheKey, data);

            return result;
        } catch (error) {
            console.error('Failed to save annotations:', error);
            throw error;
        }
    }

    // 获取下一个episode
    async getNextEpisode(episodeId) {
        try {
            return await this.request(`/api/next_episode/${episodeId}`);
        } catch (error) {
            console.error('Failed to get next episode:', error);
            throw error;
        }
    }

    // 获取视频配置
    async getVideoKeys() {
        const cacheKey = 'video_keys';

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            const data = await this.request('/api/video_keys');
            this.cache.set(cacheKey, data);
            return data;
        } catch (error) {
            console.error('Failed to get video keys:', error);
            return [];
        }
    }

    // 清除缓存
    clearCache(pattern = null) {
        if (pattern) {
            for (const key of this.cache.keys()) {
                if (key.includes(pattern)) {
                    this.cache.delete(key);
                }
            }
        } else {
            this.cache.clear();
        }
    }
}

// 工具函数
const AnnotationUtils = {
    // 时间格式化
    formatTime(time) {
        if (isNaN(time)) return '0:00';
        const hours = Math.floor(time / 3600);
        const minutes = Math.floor((time % 3600) / 60);
        const seconds = Math.floor(time % 60);
        return (hours > 0 ? hours + ':' : '') +
               (minutes < 10 ? '0' + minutes : minutes) + ':' +
               (seconds < 10 ? '0' + seconds : seconds);
    },

    // 帧范围格式化
    formatFrameRange(startFrame, endFrame) {
        const duration = endFrame - startFrame + 1;
        return `帧 ${startFrame}-${endFrame} (${duration}帧)`;
    },

    // 技能颜色映射
    getSkillColor(skill) {
        const colors = {
            'Pick': '#10B981',     // green
            'Place': '#3B82F6',    // blue
            'Move': '#8B5CF6',     // purple
            'Rotate': '#F59E0B',   // amber
            'Push': '#EF4444',     // red
            'Pull': '#EC4899',     // pink
            'Grasp': '#06B6D4',    // cyan
            'Release': '#84CC16'   // lime
        };
        return colors[skill] || '#6B7280'; // default gray
    },

    // 数据导出
    exportToJSON(data) {
        const blob = new Blob([JSON.stringify(data, null, 2)],
                            { type: 'application/json' });
        return URL.createObjectURL(blob);
    },

    // 数据验证
    validateAnnotationData(data) {
        if (!data || typeof data !== 'object') {
            return '标注数据格式无效';
        }

        if (typeof data.episode_id !== 'number' || data.episode_id < 0) {
            return 'Episode ID 无效';
        }

        if (!data.label_info || !Array.isArray(data.label_info.action_config)) {
            return '标注配置格式无效';
        }

        // 验证每个段落
        for (let i = 0; i < data.label_info.action_config.length; i++) {
            const segment = data.label_info.action_config[i];
            const error = this.validateSegment(segment, i);
            if (error) return error;
        }

        return null;
    },

    validateSegment(segment, index) {
        if (!segment || typeof segment !== 'object') {
            return `段落 ${index + 1}: 格式无效`;
        }

        const required = ['start_frame', 'end_frame', 'skill', 'action_text'];
        for (const field of required) {
            if (!(field in segment)) {
                return `段落 ${index + 1}: 缺少字段 ${field}`;
            }
        }

        if (segment.start_frame > segment.end_frame) {
            return `段落 ${index + 1}: 起始帧不能大于结束帧`;
        }

        return null;
    },

    // 调试辅助
    debugInfo(videoController, segmentManager, stateMachine) {
        return {
            video: {
                currentTime: videoController.currentTime,
                currentFrame: videoController.getCurrentFrame(),
                totalFrames: videoController.getTotalFrames(),
                isPlaying: videoController.isPlaying
            },
            segments: {
                count: segmentManager.segments.length,
                editing: segmentManager.editingIndex,
                current: segmentManager.currentSegment
            },
            state: stateMachine.getState()
        };
    }
};

// 导出到全局作用域 (用于在HTML中访问)
if (typeof window !== 'undefined') {
    window.VideoFrameCalculator = VideoFrameCalculator;
    window.VideoSyncController = VideoSyncController;
    window.AnnotationStateMachine = AnnotationStateMachine;
    window.SegmentManager = SegmentManager;
    window.KeyboardHandler = KeyboardHandler;
    window.APIClient = APIClient;
    window.AnnotationUtils = AnnotationUtils;
}

// ES6 模块导出 (如果需要)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VideoFrameCalculator,
        VideoSyncController,
        AnnotationStateMachine,
        SegmentManager,
        KeyboardHandler,
        APIClient,
        AnnotationUtils
    };
}