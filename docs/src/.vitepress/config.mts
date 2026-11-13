import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import { juliaReplTransformer } from './julia-repl-transformer'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// ビルド時に置換
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// サイト全体の設定
export default defineConfig({
  lang: 'ja-JP',
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// ビルド時に置換
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // 出力先はビルド時に決まる
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // 独自ドメインの指定は deploy_url で管理する
    ['script', {src: `${baseTemp.base}siteinfo.js`}],
    // REPLACE_ME_DOCUMENTER_VITEPRESS_NOINDEX
  ],

  markdown: {
    component: { inlineTags: ['CitationPreview'] },
    codeTransformers: [juliaReplTransformer()],
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        // Vite による処理が必要な依存関係
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },
  themeConfig: {
    outline: { level: 'deep', label: 'このページの内容' },
    docFooter: { prev: '前のページ', next: '次のページ' },
    lastUpdated: { text: '最終更新' },
    sidebarMenuLabel: 'メニュー',
    returnToTopLabel: '先頭に戻る',
    darkModeSwitchLabel: '表示モード',
    lightModeSwitchTitle: 'ライトモードに切り替える',
    darkModeSwitchTitle: 'ダークモードに切り替える',
    skipToContentLabel: '本文に移動',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        translations: {
          button: { buttonText: '検索', buttonAriaLabel: 'ドキュメントを検索' },
          modal: {
            noResultsText: '検索結果が見つかりません',
            resetButtonTitle: '検索をリセット',
            displayDetails: '詳細を表示',
            backButtonTitle: '検索を閉じる',
            footer: { selectText: '選択', navigateText: '移動', closeText: '閉じる' }
          }
        }
      }
    },
    nav,
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    sidebarDrawer: 'REPLACE_ME_DOCUMENTER_VITEPRESS_SIDEBAR_DRAWER',
    editLink: {
      pattern: 'https://github.com/AtelierArith/Colmat.jl/edit/main/docs/src/:path',
      text: 'このページを編集'
    },
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: '作成ツール: <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© ${new Date().getUTCFullYear()}.`
    }
  }
})
