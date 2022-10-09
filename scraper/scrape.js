import fs from "fs-extra"
import path from "path"
import process from "process"
import yargs from "yargs";
import {
    hideBin
} from "yargs/helpers";
import {
    JSDOM
} from "jsdom"
const page = "https://www.amazon.com/s?i=electronics-intl-ship&rh=n%3A16225009011&fs=true&page=";
const OUTPUT_FOLDER = "reviews";
await fs.ensureDir(OUTPUT_FOLDER);
const args = yargs(hideBin(process.argv)).option("page-start", {
        type: "number",
        description: "Define the first page of products that is to be scraped. Pages before that will be skipped.",
    }).option("max-pages", {
        type: "number",
        description: "Define how many extra pages after the first one, the program should loop through.",
    })
    .strict().demandCommand(2).recommendCommands().argv;
const START_PAGE = args._[0];
const MAX_PAGES = START_PAGE + args._[1];
for (let i = START_PAGE; i < MAX_PAGES; i++) {
    console.log(`Getting products from page ${i}`);
    const response = await fetch(page + i);
    const dom = new JSDOM(await response.text());
    const productReviewLinks = [...dom.window.document.querySelectorAll(".s-main-slot .s-product-image-container a[href]")].map(link => {
        return link.href.replace("/dp/", "/product-reviews/");
    });
    for (const productReviewLink of productReviewLinks) {
        const id = productReviewLink.match(/\/(?<id>[^\/]{10})\//).groups.id;
        const filepath = path.join(OUTPUT_FOLDER, `${id}.csv`);
        process.stdout.write(`Getting reviews for ${id}, page:`);
        if (await fs.exists(filepath)) {
            console.log(`\nProduct file "${filepath}" already exists, skipping`);
            continue;
        }
        let finalContents = "rating,review\n";
        let noMoreReviews = false;
        for (let n = 1; !noMoreReviews; n++) {
            process.stdout.write(` ${n}`);
            const response1 = await fetch(`https://www.amazon.com${productReviewLink}&pageNumber=${n}`);
            const dom1 = new JSDOM(await response1.text());
            const reviews = [...dom1.window.document.querySelectorAll(`.reviews-content [id^="customer_review"]`)];
            for (const review of reviews) {
                const reviewText = review.querySelector("span.review-text").textContent.replaceAll(/[\n\r]/g, '').replaceAll(",", "").trim();
                const ratingElement = review.querySelector(".review-rating");
                let rating;
                for (const ratingOpt of [0, 1, 2, 3, 4, 5]) {
                    if (ratingElement.classList.contains(`a-star-${ratingOpt}`)) {
                        rating = ratingOpt;
                        break;
                    }
                }
                finalContents += `${rating},${reviewText}\n`;

            }
            if (reviews.length === 0) noMoreReviews = true;
        }
        process.stdout.write("\n");
        await fs.writeFile(filepath, finalContents);
    }
}
