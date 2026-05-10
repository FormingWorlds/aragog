/* Open external links in a new tab.
 *
 * Any anchor whose href is an absolute http(s) URL pointing at a host that
 * differs from the current site host gets target="_blank" and the
 * rel="noopener noreferrer" hardening. Internal in-site links and intra-page
 * fragment / mailto: / tel: links are left alone, so navigation between docs
 * pages stays in the same tab. */
(function () {
  function applyExternalLinkBehavior() {
    var here = window.location.host;
    var anchors = document.querySelectorAll('a[href]');
    for (var i = 0; i < anchors.length; i++) {
      var a = anchors[i];
      var href = a.getAttribute('href');
      if (!href) continue;
      // Only act on absolute http(s) URLs — skip relative, fragment, mailto:, tel:, etc.
      if (!/^https?:\/\//i.test(href)) continue;
      try {
        var url = new URL(href);
        if (url.host && url.host !== here) {
          a.setAttribute('target', '_blank');
          a.setAttribute('rel', 'noopener noreferrer');
        }
      } catch (e) {
        // Malformed URL — ignore.
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyExternalLinkBehavior);
  } else {
    applyExternalLinkBehavior();
  }
})();
