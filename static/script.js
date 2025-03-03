document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("header nav ul li a").forEach(link => {
        link.addEventListener("click", function (event) {
            event.preventDefault(); // Prevent default anchor behavior
            const page = this.getAttribute("data-page");
            window.location.href = page;
        });
    });
});
