// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="index.html"><strong aria-hidden="true">1.</strong> Home</a></li><li class="chapter-item expanded "><a href="day-01.html"><strong aria-hidden="true">2.</strong> Day 01 關於這次鐵人賽…前言以及規劃</a></li><li class="chapter-item expanded "><a href="day-02.html"><strong aria-hidden="true">3.</strong> Day 02 最佳化演算法的5W1H</a></li><li class="chapter-item expanded "><a href="day-03.html"><strong aria-hidden="true">4.</strong> Day 03 當今的最佳化手段有甚麼？</a></li><li class="chapter-item expanded "><a href="day-04.html"><strong aria-hidden="true">5.</strong> Day 04 最佳化演算法入門，更深入的了解啟發式演算法！</a></li><li class="chapter-item expanded "><a href="day-05.html"><strong aria-hidden="true">6.</strong> Day 05 如何知道最佳化演算法的優劣？測試函數介紹</a></li><li class="chapter-item expanded "><a href="day-06.html"><strong aria-hidden="true">7.</strong> Day 06 根據方程式來寫出測試函數的程式吧！(1/3)</a></li><li class="chapter-item expanded "><a href="day-07.html"><strong aria-hidden="true">8.</strong> Day 07 根據方程式來寫出測試函數的程式吧！(2/3)</a></li><li class="chapter-item expanded "><a href="day-08.html"><strong aria-hidden="true">9.</strong> Day 08 根據方程式來寫出測試函數的程式吧！(3/3)</a></li><li class="chapter-item expanded "><a href="day-09.html"><strong aria-hidden="true">10.</strong> Day 09 目前Python最佳化演算法的函式庫有啥？</a></li><li class="chapter-item expanded "><a href="day-10.html"><strong aria-hidden="true">11.</strong> Day 10 基於粒子(swarm-based)的啟發式演算法是甚麼？</a></li><li class="chapter-item expanded "><a href="day-11.html"><strong aria-hidden="true">12.</strong> Day 11 基於進化(evolutionary-based)的啟發式演算法是甚麼？</a></li><li class="chapter-item expanded "><a href="day-12.html"><strong aria-hidden="true">13.</strong> Day 12 基於人類行為(human_based)的啟發式演算法是甚麼？</a></li><li class="chapter-item expanded "><a href="day-13.html"><strong aria-hidden="true">14.</strong> Day 13 基於生物學(biology-based)的啟發式演算法是甚麼？</a></li><li class="chapter-item expanded "><a href="day-14.html"><strong aria-hidden="true">15.</strong> Day 14 無痛入門！淺談Optuna最佳化</a></li><li class="chapter-item expanded "><a href="day-15.html"><strong aria-hidden="true">16.</strong> Day 15 由淺入深！介紹更多Optuna的API(1/2)</a></li><li class="chapter-item expanded "><a href="day-16.html"><strong aria-hidden="true">17.</strong> Day 16 由淺入深！介紹更多Optuna的API (2/2)</a></li><li class="chapter-item expanded "><a href="day-17.html"><strong aria-hidden="true">18.</strong> Day 17 打鐵趁熱！來試著使用Optuna解決問題吧</a></li><li class="chapter-item expanded "><a href="day-18.html"><strong aria-hidden="true">19.</strong> Day 18 Optuna的背後演算法，TPE介紹</a></li><li class="chapter-item expanded "><a href="day-19.html"><strong aria-hidden="true">20.</strong> Day 19 Optuna的更多應用，最佳化MLP與CNN網路</a></li><li class="chapter-item expanded "><a href="day-20.html"><strong aria-hidden="true">21.</strong> Day 20 Optuna的更多應用，最佳化生成對抗網路(GAN)(1/2)</a></li><li class="chapter-item expanded "><a href="day-21.html"><strong aria-hidden="true">22.</strong> Day 21 Optuna的更多應用，最佳化生成對抗網路(GAN)(2/2)</a></li><li class="chapter-item expanded "><a href="day-22.html"><strong aria-hidden="true">23.</strong> Day 22 無痛入門！淺談MealPy最佳化</a></li><li class="chapter-item expanded "><a href="day-23.html"><strong aria-hidden="true">24.</strong> Day 23 由淺入深！介紹更多MealPy的API (1/2)</a></li><li class="chapter-item expanded "><a href="day-24.html"><strong aria-hidden="true">25.</strong> Day 24 由淺入深！介紹更多MealPy的API (2/2)</a></li><li class="chapter-item expanded "><a href="day-25.html"><strong aria-hidden="true">26.</strong> Day 25 打鐵趁熱！來試著使用MealPy解決問題吧</a></li><li class="chapter-item expanded "><a href="day-26.html"><strong aria-hidden="true">27.</strong> Day 26 打鐵趁熱！來試著MealPy的更多應用，最佳化MLP與CNN網路</a></li><li class="chapter-item expanded "><a href="day-27.html"><strong aria-hidden="true">28.</strong> Day 27 MealPy的更多應用，最佳化生成對抗網路(GAN)(1/2)</a></li><li class="chapter-item expanded "><a href="day-28.html"><strong aria-hidden="true">29.</strong> Day 28 MealPy的更多應用，最佳化生成對抗網路(GAN)(2/2)</a></li><li class="chapter-item expanded "><a href="day-29.html"><strong aria-hidden="true">30.</strong> Day 29 關於其他機器學習與深度學習的最佳化應用...</a></li><li class="chapter-item expanded "><a href="day-30.html"><strong aria-hidden="true">31.</strong> Day 30 關於這次鐵人賽！總結與心得</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
