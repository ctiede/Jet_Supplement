;; .emacs

;;; uncomment this line to disable loading of "default.el" at startup
;; (setq inhibit-default-init t)



;; turn on font-lock mode
(when (fboundp 'global-font-lock-mode)
  (global-font-lock-mode t))

;; enable visual feedback on selections
;(setq transient-mark-mode t)

;; default to better frame titles
(setq frame-title-format
      (concat  "%b - emacs@" (system-name)))

;; default to unified diffs
(setq diff-switches "-u")

;; always end a file with a newline
;(setq require-final-newline 'query)


(setq mouse-wheel-mode t)                                                       
(transient-mark-mode t)                                                         
(global-set-key (kbd "RET") 'newline-and-indent)     

(setq column-number-mode t)                                                     
(setq x-select-enable-clipboard t)       

;;Display matching parentheses when cursor is near one of them
(require 'paren)(show-paren-mode t)


(add-to-list 'load-path "~/local/ecb-master")
(autoload 'ecb "ecb.el")
;;(require 'ecb)
(setq stack-trace-on-error t)
(setq ecb-show-sources-in-directories-buffer 'always)
(setq auto-expand-directory-tree 'best)    
(setq ecb-layout-name "left3")     

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-safe-themes (quote ("9b94a52c25ea76b72df2050928d18e7fe9060e9c7f7d992f33bf35d4931b0444" "de538b2d1282b23ca41ac5d8b69c033b911521fe27b5e1f59783d2eb20384e1f" default)))
 '(ecb-options-version "2.40"))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )


(custom-set-variables
  ;; custom-set-variables was added by Custom.
  ;; If you edit it by hand, you could mess it up, so be careful.
  ;; Your init file should contain only one such instance.
  ;; If there is more than one, they won't work right.
 '(ecb-layout-window-sizes
   (quote
    (("left3"
      (ecb-directories-buffer-name 0.2100 . 0.29545454545454547)
      (ecb-sources-buffer-name 0.2100 . 0.3409090909090909)
      (ecb-methods-buffer-name 0.2100 . 0.3409090909090909))
     ("leftright2"
      (ecb-directories-buffer-name 0.1025 . 0.6285714285714286)
      (ecb-sources-buffer-name 0.1025 . 0.34285714285714286)
      (ecb-methods-buffer-name 0.1125 . 0.6285714285714286)
      (ecb-history-buffer-name 0.1125 . 0.34285714285714286)))))
 '(ecb-options-version "2.40"))
                                  

(define-key input-decode-map "\e[1;2A" [S-up])
(define-key input-decode-map "\e[1;2B" [S-down])

(when (fboundp 'windmove-default-keybindings)
    (windmove-default-keybindings))

;; (defun ignore-error-wrapper (fn)
;;     "Funtion return new function that ignore errors.
;;    The function wraps a function with `ignore-errors' macro."
;;       (lexical-let ((fn fn))
;; 	    (lambda ()
;; 	            (interactive)
;; 		          (ignore-errors
;; 			            (funcall fn)))))

;; (global-set-key [s-left] (ignore-error-wrapper 'windmove-left))
;; (global-set-key [s-right] (ignore-error-wrapper 'windmove-right))
;; (global-set-key [s-up] (ignore-error-wrapper 'windmove-up))
;; (global-set-key [s-down] (ignore-error-wrapper 'windmove-down))    

(require 'package) ;; You might already have this line
(add-to-list 'package-archives
	                  '("melpa-stable" . "http://melpa-stable.milkbox.net/packages/") t)
(when (< emacs-major-version 24)
    ;; For important compatibility libraries like cl-lib
    (add-to-list 'package-archives '("gnu" . "http://elpa.gnu.org/packages/")))
(package-initialize) ;; You might already have this line

(defadvice package-compute-transaction  (before package-compute-transaction-reverse (package-list requirements) activate compile)
    "reverse the requirements"
      (setq requirements (reverse requirements))
        (print requirements))

(load-theme 'zenburn t)    
(require 'ido)
(ido-mode t)
(put 'upcase-region 'disabled nil)
(which-func-mode t)

(require 'hl-line)
;; (global-hl-line-mode)
;; (set-face-background hl-line-face "color-105") ;; list-colors-display show all color
(set-face-attribute hl-line-face nil :underline t) ;; list-colors-display show all color

(put 'set-goal-column 'disabled nil)
(put 'scroll-left 'disabled nil)


;; (defun copy-from-osx ()
;;     (shell-command-to-string "xclip"))

;; (defun paste-to-osx (text &optional push)
;;     (let ((process-connection-type nil))
;;           (let ((proc (start-process "xclip" "*Messages*" "xclip")))
;; 	          (process-send-string proc text)
;; 		        (process-send-eof proc))))

;; (setq interprogram-cut-function 'paste-to-osx)
;; (setq interprogram-paste-function 'copy-from-osx)

