/*!
 * Live2D Widget
 * https://github.com/stevenjoezhang/live2d-widget
 */

// Resolve bundled widget assets relative to this vendored script.
const live2d_path = new URL('./', import.meta.url).href;

// Method to encapsulate asynchronous resource loading
// 封装异步加载资源的方法
function loadExternalResource(url, type) {
  return new Promise((resolve, reject) => {
    let tag;

    if (type === 'css') {
      tag = document.createElement('link');
      tag.rel = 'stylesheet';
      tag.href = url;
    }
    else if (type === 'js') {
      tag = document.createElement('script');
      tag.type = 'module';
      tag.src = url;
    }
    if (tag) {
      tag.onload = () => resolve(url);
      tag.onerror = () => reject(url);
      document.head.appendChild(tag);
    }
  });
}

function pickNextTextureId(textureIds, currentTextureId) {
  const candidates = textureIds.filter((textureId) => textureId !== currentTextureId);
  return candidates[Math.floor(Math.random() * candidates.length)];
}

function forceNextSwitchTexture(modelId, textureIds, modelTextureCount) {
  document.addEventListener('click', (event) => {
    if (!(event.target instanceof Element)) return;
    if (!event.target.closest('#waifu-tool-switch-texture')) return;
    if (Number(localStorage.getItem('modelId')) !== modelId) return;
    if (textureIds.length < 2) return;

    const currentTextureId = Number(localStorage.getItem('modelTexturesId'));
    const nextTextureId = pickNextTextureId(textureIds, currentTextureId);
    const sampledIndex = nextTextureId > currentTextureId ? nextTextureId - 1 : nextTextureId;
    const forcedRandom = (sampledIndex + 0.5) / (modelTextureCount - 1);
    const originalRandom = Math.random;
    let used = false;

    const forcedRandomOnce = function(...args) {
      if (!used) {
        used = true;
        return forcedRandom;
      }
      return originalRandom.apply(this, args);
    };
    Math.random = forcedRandomOnce;
    setTimeout(() => {
      if (Math.random === forcedRandomOnce) Math.random = originalRandom;
    }, 0);
  }, true);
}

(async () => {
  // If you are concerned about display issues on mobile devices, you can use screen.width to determine whether to load
  // 如果担心手机上显示效果不佳，可以根据屏幕宽度来判断是否加载
  // if (screen.width < 768) return;

  // Avoid cross-origin issues with image resources
  // 避免图片资源跨域问题
  const OriginalImage = window.Image;
  window.Image = function(...args) {
    const img = new OriginalImage(...args);
    img.crossOrigin = "anonymous";
    return img;
  };
  window.Image.prototype = OriginalImage.prototype;
  // Load waifu.css and waifu-tips.js
  // 加载 waifu.css 和 waifu-tips.js
  await Promise.all([
    loadExternalResource(live2d_path + 'waifu.css', 'css'),
    loadExternalResource(live2d_path + 'waifu-tips.js', 'js')
  ]);
  const defaultModelId = 2;
  const defaultTextureId = 15;
  const switchTextureIds = [7, 8, 13, 14, 15, 16, 17];
  const defaultModelTextureCount = 20;
  localStorage.setItem('modelId', String(defaultModelId));
  localStorage.setItem('modelTexturesId', String(defaultTextureId));
  forceNextSwitchTexture(defaultModelId, switchTextureIds, defaultModelTextureCount);

  // For detailed usage of configuration options, see README.en.md
  // 配置选项的具体用法见 README.md
  initWidget({
    waifuPath: live2d_path + 'waifu-tips.json',
    // cdnPath: 'https://fastly.jsdelivr.net/gh/fghrsh/live2d_api/',
    cubism2Path: live2d_path + 'live2d.min.js',
    cubism5Path: 'https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js',
    tools: ['switch-model', 'switch-texture', 'photo', 'info', 'quit'],
    logLevel: 'warn',
    drag: true,
  });
})();

console.log(`\n%cLive2D%cWidget%c\n`, 'padding: 8px; background: #cd3e45; font-weight: bold; font-size: large; color: white;', 'padding: 8px; background: #ff5450; font-size: large; color: #eee;', '');

/*
く__,.ヘヽ.        /  ,ー､ 〉
         ＼ ', !-─‐-i  /  /´
         ／｀ｰ'       L/／｀ヽ､
       /   ／,   /|   ,   ,       ',
     ｲ   / /-‐/  ｉ  L_ ﾊ ヽ!   i
      ﾚ ﾍ 7ｲ｀ﾄ   ﾚ'ｧ-ﾄ､!ハ|   |
        !,/7 '0'     ´0iソ|    |
        |.从"    _     ,,,, / |./    |
        ﾚ'| i＞.､,,__  _,.イ /   .i   |
          ﾚ'| | / k_７_/ﾚ'ヽ,  ﾊ.  |
            | |/i 〈|/   i  ,.ﾍ |  i  |
           .|/ /  ｉ：    ﾍ!    ＼  |
            kヽ>､ﾊ    _,.ﾍ､    /､!
            !'〈//｀Ｔ´', ＼ ｀'7'ｰr'
            ﾚ'ヽL__|___i,___,ンﾚ|ノ
                ﾄ-,/  |___./
                'ｰ'    !_,.:
*/
